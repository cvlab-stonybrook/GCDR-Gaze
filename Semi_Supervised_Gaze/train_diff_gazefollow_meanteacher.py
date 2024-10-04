import os, sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import pdb
import traceback
import torch.nn.functional as F
from tqdm import tqdm
from diffusion.gaze_diff_model import GF_Diffusion_Model_new
from utils import imutils, evaluation, myutils
from torchvision import transforms
from utils.torchtools import save_checkpoint
from datasets.dataset_weakly import GazeFollow_weakly_doubleaug
from weak_sup.semi_utils import update_ema_variables, get_current_consistency_weight, get_label_idx, TwoStreamBatchSampler

def get_transform(input_resolution):
    
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='0', help="gpu id")
    parser.add_argument('--project_name', default='diffusion_meanteacher')
    parser.add_argument('--setting_name', default='')
    parser.add_argument('--onlyin', action='store_true')
    parser.add_argument('--config', default='./config/config_diffusion.yaml')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--no_inout', action='store_true')
    parser.add_argument('--schedule', type=str, default='linear')
    parser.add_argument('--hm_sigma', type=float, default=3.0, help='sigma value for creating the ground truth heatmap')
    parser.add_argument('--optim', default='adam')
    parser.add_argument('--plot_every', type=int, default=400)
    parser.add_argument('--coord_regression', action='store_true')
    parser.add_argument('--loss_amp_factor', type=float, default=10000)
    parser.add_argument('--lambda_', type=float, default=40.0)
    parser.add_argument("--lr", type=float, default=2.5e-5, help="learning rate")
    parser.add_argument('--pred_noise', action='store_true')
    parser.add_argument('--consistency', type=float, default=4000.0)
    parser.add_argument('--consistency_rampup', type=int, default=5)
    parser.add_argument('--consistency_type', default="mse", type=str, metavar='TYPE',
                        choices=['mse', 'kl'],
                        help='consistency loss type to use')
    parser.add_argument('--labeled-batch-size', default=None, type=int,
                        metavar='N', help="labeled examples per minibatch (default: no constrain)")
    parser.add_argument('--ema-decay', default=0.99, type=float, metavar='ALPHA',
                        help='ema variable decay rate (default: 0.999)')
    parser.add_argument('--alpha', type=float, default=100.0, help='weights to apply on the heatmap loss')
    parser.add_argument("--batch_size", type=int, default=80, help="batch size")
    parser.add_argument("--epochs", type=int, default=150, help="number of epochs")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay")
    parser.add_argument('--supervise_ratio', type=float, default=1.0, help='ratio of data with ground truth annotation in semi-supervised training')
    parser.add_argument("--print_every", type=int, default=100, help="print every ___ iterations")
    parser.add_argument("--save_every", type=int, default=1, help="save every ___ epochs")
    parser.add_argument("--eval_every", type=int, default=1, help="evaluate every ___ epochs")
    parser.add_argument("--resume", type=int, default=-1, help="which epoch to resume from")
    parser.add_argument('--remark', type=str, default='')
    parser.add_argument('--scale', type=float, default=2.0)
    parser.add_argument('--scale_input', action='store_true')
    parser.add_argument('--no_ddim', dest='ddim', action='store_false')
    parser.add_argument('--ddim_eta', type=float, default=1.0)
    parser.add_argument('--inference_times', type=int, default=1) 
    parser.add_argument('--inference_steps', type=int, default=20, help='average the sampled outputs during inference as the final output')
    parser.add_argument('--time_steps', type=int, default=500, help='steps in diffusion training')
    parser.add_argument("--init_weights", type=str, default="/nfs/bigcortex/add_disk0/qiaomu/ckpts/gaze/videoatttarget/initial_weights_for_spatial_training.pt", help="initial weights")
    parser.add_argument("--ckpt_dir", type=str, default="/nfs/bigrod/add_disk0/qiaomu/ckpts/gaze/gazefollow_semisup", help="directory to save log files")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    cfg = myutils.update_config(args.config)
    for k in cfg.__dict__:
        setattr(args, k, cfg[k])
    
    args.MODEL.DIFF_MODEL.inference_steps = args.inference_steps
    args.MODEL.DIFF_MODEL.pred_noise = args.pred_noise
    args.MODEL.DIFF_MODEL.time_steps = args.time_steps
    args.MODEL.DIFF_MODEL.denoise_steps = -1
    args.MODEL.DIFF_MODEL.scale = args.scale
    args.MODEL.DIFF_MODEL.scale_input = args.scale_input
    args.MODEL.DIFF_MODEL.noise_schedule = args.schedule
    args.MODEL.DIFF_MODEL.ddim = args.ddim
    args.MODEL.DIFF_MODEL.ddim_eta = args.ddim_eta
    np.random.seed(1)
    input_resolution, output_resolution = args.DATA.input_resolution, args.DATA.output_resolution
    transform = get_transform(input_resolution)

    train_dataset = GazeFollow_weakly_doubleaug(args.DATA, transform, test=False, ratio = args.supervise_ratio, use_sup_only=False)
    val_dataset = GazeFollow_weakly_doubleaug(args.DATA, transform, test=True)

    labeled_idxs, unlabeled_idxs = get_label_idx(train_dataset)
    
    batch_sampler = TwoStreamBatchSampler(
        unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_sampler = batch_sampler,
                                                num_workers=14)
        
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                batch_size=args.batch_size*2,
                                                shuffle=False,
                                                num_workers=16)

    model = GF_Diffusion_Model_new(args.MODEL)
    teacher_model = GF_Diffusion_Model_new(args.MODEL)
    myutils.load_pretrained_weights(model.feat_extractor, state_dict = None, weight_path=args.init_weights) 
    myutils.load_pretrained_weights(teacher_model.feat_extractor, state_dict = None, weight_path=args.init_weights) 
    model.cuda()
    teacher_model.cuda()


    mse_loss = nn.MSELoss(reduction='none') # not reducing in order to ignore outside cases
    kl_div = nn.KLDivLoss(reduction='none')

    if args.consistency_type == 'mse':
        consistency_criterion = mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = kl_div


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mse_loss = nn.MSELoss(reduction='none') # not reducing in order to ignore outside cases
    mseloss_mean = nn.MSELoss(reduction='mean')

    if args.resume==-1:
        setting_name = f"{args.remark}diffmodel_steps{args.time_steps}_infsteps{args.inference_steps}_ratio{args.supervise_ratio}_type{args.consistency_type}_consist{args.consistency}_ramp{args.consistency_rampup}_ema{args.ema_decay}_sche{args.schedule}_scale{args.scale}_lr{args.lr}bs{args.batch_size}_onlyin{args.onlyin}_ampfactor{args.loss_amp_factor}_weightdecay{args.weight_decay}"
    else:
        setting_name = args.setting_name
    logdir, ckpt_dir, logger, writer = myutils.setup_logger_tensorboard("semi_sup_gf/"+args.project_name, setting_name, args.ckpt_dir, resume=args.resume)
    logger.info(args)
    step=0
    max_steps = len(train_loader)
    multigpu = False
    if len(args.device)>1:
        multigpu=True
        model = nn.DataParallel(model)

    for param in teacher_model.parameters():
        param.detach_()

    if args.resume!=-1:
        ckpt_load = os.path.join(args.ckpt_dir, args.project_name, setting_name, 'epoch_%02d_weights.pt' % (args.resume))
        torch_load = torch.load(ckpt_load)
        state_dict = torch_load['state_dict']
        myutils.load_pretrained_weights(model, state_dict = state_dict, weight_path=ckpt_load) 
        print("successfully load {}!".format(ckpt_load))
        step = torch_load['train_step']
    eps=1e-11


    for ep in range(max(args.resume+1,0), args.epochs):
        
        model.train(True)
        teacher_model.train(True)
        ep_loss, ep_l2_loss, ep_consist_loss = 0.0, 0.0, 0.0
        consistency_weight = get_current_consistency_weight(args.consistency, ep, args.consistency_rampup)
        logger.info(f"Epoch {ep}: learning rate: {optimizer.param_groups[0]['lr']}, consistency weight: {consistency_weight}") 
        num_sup, num_all = 0,0

        for batch, (img_1, face_1, head_channel_1, img_2, face_2, head_channel_2, gaze_heatmap, gaze_inside, head_coords, body_coords, gaze_coords, gradcam_resize, sup, path) in enumerate(train_loader):
            images_1, head_1, faces_1 = img_1.cuda(), head_channel_1.cuda(), face_1.cuda()
            images_2, head_2, faces_2 = img_2.cuda(), head_channel_2.cuda(), face_2.cuda()
            head_coords, gaze_coords = head_coords.float().cuda(), gaze_coords.float().cuda()
            gaze_heatmap = gaze_heatmap.cuda()
            gradcam_resize = gradcam_resize.cuda()
            gaze_inside = gaze_inside.cuda().to(torch.float)
            sup = sup.cuda()
            
            sup_mask = sup.bool()
            
            esp_est_t = teacher_model.forward_inference([images_2, head_2, faces_2], 10).squeeze(1)
            esp_est, epsilon, t = model.denoise_with_unsup([images_1, head_1, faces_1, gaze_heatmap.unsqueeze(1)], sup_mask)
            esp_est, epsilon = esp_est.squeeze(1), epsilon.squeeze(1)
            
            if args.consistency>0:
                # consistency loss is applied on both supervised and unsupervised data
                sup_inside = torch.logical_and(gaze_inside.bool(), sup_mask)
                unsup_inside = torch.logical_and(gaze_inside.bool(), ~sup_mask)
                inside_mask = torch.logical_or(sup_inside, unsup_inside).float()
                valid_samples = torch.sum(inside_mask)
                input_x, input_y = esp_est, esp_est_t
                
                if args.consistency_type=='kl':
                    input_x = F.log_softmax(input_x.flatten(start_dim=1), dim=1)
                    input_y = F.softmax(input_y.flatten(start_dim=1), dim=1)
                    
                consistency_loss = consistency_criterion(input_x, input_y) # only for supervised labels
                if args.consistency_type=='mse':
                    consistency_loss = torch.mean(consistency_loss, dim=1)
                    consistency_loss = torch.mean(consistency_loss, dim=1)
                else:
                    consistency_loss = torch.sum(consistency_loss, dim=1)
                
                consistency_loss = torch.mul(consistency_loss, inside_mask)
                consistency_loss = torch.sum(consistency_loss)/valid_samples
                consistency_loss = torch.mean(consistency_loss)
                consistency_loss = consistency_loss * consistency_weight 
            
            
            if not args.onlyin:
                l2_loss = mseloss_mean(esp_est[sup_mask], gaze_heatmap[sup_mask])
            else:
                if args.pred_noise:
                    l2_loss = mse_loss(esp_est[sup_mask],epsilon[sup_mask])
                else:
                    # add scale gaze heatmap
                    l2_loss = mse_loss(esp_est[sup_mask], gaze_heatmap[sup_mask])  # let the model predict gt directly
                l2_loss = torch.mean(l2_loss, dim=1)
                l2_loss = torch.mean(l2_loss, dim=1)
                l2_loss = torch.mul(l2_loss, gaze_inside[sup_mask])
                l2_loss = torch.sum(l2_loss)/torch.sum(gaze_inside[sup_mask])
            
            l2_loss = l2_loss * args.loss_amp_factor
            num_in = gaze_inside.sum().item()
            num_out = img_1.size()[0]-num_in
            num_sup += sup.sum().item()
            num_all += sup.size()[0]
            
            total_loss = l2_loss + consistency_loss
            writer.add_scalar('train/Loss_step', total_loss.item(), step)
            writer.add_scalar('train/l2_loss_step', l2_loss.item(), step)
            writer.add_scalar('train/Consistency_loss_step', consistency_loss.item(), step)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            update_ema_variables(model, teacher_model, args.ema_decay, step//15)
            
            if batch % args.print_every == 0:
                print("Epoch:{:04d}\tstep:{:06d}/{:06d}\ttraining loss: (l2){:.4f} (Consistency) {:.4f}".format(ep, batch+1, max_steps, l2_loss, consistency_loss))
                sup_ratio = num_sup / num_all
                print("number inside: {}, number outside: {}, supervise_ratio:{:.2f}%".format(num_in, num_out, sup_ratio*100))

            step += 1

        
        if not args.debug and ep % args.eval_every==0:
            logger.info('Validation in progress ...: epoch {}'.format(ep))
            model.train(False)
            AUC_avg = []; min_dist_avg = []; avg_dist_avg = []; in_vs_out_groundtruth = []; in_vs_out_pred = []
            ep_val_loss, ep_val_l2_loss, ep_val_inout_loss, ep_val_coord_loss = 0.0, 0.0, 0.0, 0.0
            num_imgs = len(val_dataset)
            #select_idx = random.randint(0, num_imgs-1)
            select_idx = 0
            start_idx, end_idx = 0,0
            model.eval()
            with torch.no_grad():
                AUC = []; min_dist = []; avg_dist = []
                all_batches = 0
                for eval_iter in range(args.inference_times):
                    for val_batch, (img, face, head_channel, gaze_heatmap, cont_gaze, imsize, head_coords, body_coords, gaze_coords, gradcam_resize, path) in enumerate(tqdm(val_loader)):
                        start_idx, end_idx = end_idx, start_idx + img.size()[0]
                        images, head, faces = img.cuda(), head_channel.cuda(), face.cuda()
                        head_coords, gaze_coords = head_coords.float().cuda(), gaze_coords.float().cuda()
                        gaze_heatmap = gaze_heatmap.cuda() 
                        gaze_heatmap_pred = model([images, head, faces], args.inference_steps)
                        gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1)
                        bs = img.size()[0]
                        all_batches+=1
                        
                        # un-scale the prediction
                        #gaze_heatmap_pred = ((gaze_heatmap_pred + 1) / 2.).clamp(0.0, 1.0)
                        l2_loss = mseloss_mean(gaze_heatmap_pred, gaze_heatmap)*args.loss_amp_factor
                        gaze_inside = torch.ones(images.size()[0]).cuda()
                        total_loss = l2_loss
                        
                        ep_val_l2_loss += l2_loss.item()
                        ep_val_loss += total_loss.item()
                        
                        max_all, min_all = gaze_heatmap_pred.flatten(start_dim=1).max(dim=1)[0].unsqueeze(-1).unsqueeze(-1), gaze_heatmap_pred.flatten(start_dim=1).min(dim=1)[0].unsqueeze(-1).unsqueeze(-1)
                        gaze_heatmap_pred = (gaze_heatmap_pred - min_all) / (max_all - min_all + eps)
                        gaze_heatmap_pred = gaze_heatmap_pred.cpu().numpy()
                        gaze_heatmap = gaze_heatmap.squeeze(1).cpu().numpy()
                        
                        for b_i in range(len(cont_gaze)):
                            # remove padding and recover valid ground truth points
                            valid_gaze = cont_gaze[b_i]
                            valid_gaze = valid_gaze[valid_gaze != -1].view(-1,2)
                            valid_gaze = valid_gaze.numpy()  
                            # AUC: area under curve of ROC
                            multi_hot = imutils.multi_hot_targets(cont_gaze[b_i], imsize[b_i])
                            try:
                                auc_score, avg_dist_this, min_dist_this = evaluation.get_all_metrics(gaze_heatmap_pred[b_i], multi_hot, valid_gaze, imsize[b_i][1], imsize[b_i][0], output_resolution=64)
                            except Exception:
                                print(traceback.format_exc())
                                pdb.set_trace()
                            
                            AUC.append(auc_score)
                            min_dist.append(min_dist_this)
                            avg_dist.append(avg_dist_this)
                        
                    print("Iter{}:\tAUC:{:.4f}\tmin dist:{:.4f}\tavg dist:{:.4f}\t".format(
                        eval_iter,
                        torch.mean(torch.tensor(AUC)),
                        torch.mean(torch.tensor(min_dist)),
                        torch.mean(torch.tensor(avg_dist)),
                        ))
            
                AUC_avg.append(torch.mean(torch.tensor(AUC)))
                min_dist_avg.append(torch.mean(torch.tensor(min_dist)))
                avg_dist_avg.append(torch.mean(torch.tensor(avg_dist)))
            
            logger.info("Epoch {} averaged evaluation:\tAUC:{:.4f}\tmin dist:{:.4f}\tavg dist:{:.4f}\t".format(
                        ep,
                        torch.mean(torch.tensor(AUC)),
                        torch.mean(torch.tensor(min_dist)),
                        torch.mean(torch.tensor(avg_dist)),
                        ))
            ep_val_loss, ep_val_l2_loss, ep_val_inout_loss = ep_val_loss/(all_batches+1), ep_val_l2_loss/(all_batches+1), ep_val_inout_loss/(all_batches+1)
            
            AUC_avg = torch.mean(torch.tensor(AUC_avg))
            min_dist_avg = torch.mean(torch.tensor(min_dist_avg))
            avg_dist_avg = torch.mean(torch.tensor(avg_dist_avg))
            
            writer.add_scalar('Validation AUC', AUC_avg, global_step=ep)
            writer.add_scalar('Validation min dist', min_dist_avg, global_step=ep)
            writer.add_scalar('Validation avg dist', avg_dist_avg, global_step=ep)
            writer.add_scalar('val/Loss', ep_val_loss, ep)
            writer.add_scalar('val/l2_loss', ep_val_l2_loss, ep)

            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], ep)

            sup_ratio = num_sup / num_all
            
            if ep % args.save_every == 0 and not args.debug:
                # save the model
                if multigpu:
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                checkpoint = {'state_dict': state_dict, 'lr':optimizer.param_groups[0]['lr'], 'train_step':step}
                save_checkpoint(checkpoint, ckpt_dir, 'epoch_%02d_weights.pt' % (ep), remove_module_from_keys=True)

        writer.flush()
    writer.close()