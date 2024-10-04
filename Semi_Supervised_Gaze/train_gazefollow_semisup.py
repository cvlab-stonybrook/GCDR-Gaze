import torch
import argparse
import os
import pdb
import random
import shutil
import numpy as np
import torch.nn as nn
from skimage.transform import resize
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from utils.log_utils import get_logger
from utils import imutils, evaluation, myutils
from torchvision import transforms
from datasets.dataset_weakly import GazeFollow_weakly
from diffusion.gaze_diff_model import GF_Diffusion_Model_new
from utils.torchtools import save_checkpoint
from model.model_weakly import ModelSpatial, ModelSpatial_GradCAM
from weak_sup.semi_utils import get_label_idx, TwoStreamBatchSampler
eps = 1e-8

def l2_hm_loss(pred_hm, gt_hm, loss_apply_idx, onlyin=True):
    # all in size of bs x hm_h x hm_w
    loss_apply_onehot = loss_apply_idx.float()
    mse_loss = F.mse_loss(pred_hm, gt_hm, reduction='none')
    mse_loss = torch.mean(mse_loss, dim=1)
    loss = torch.mean(mse_loss, dim=1)
    if onlyin:
        loss = torch.mul(loss, loss_apply_onehot)
        loss = loss.sum() / loss_apply_onehot.sum()
    else:
        loss = loss.mean()
    return loss

def get_transform(input_resolution):
    
    transform_list = [] 
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)

def train(args):

    # Prepare data
    print("Loading Data")
    # Set up log dir
    cfg = myutils.update_config(args.config)
    for k in cfg.__dict__:
        setattr(args, k, cfg[k])
    args.MODEL.DIFF_MODEL.inference_steps = args.inference_steps
    args.MODEL.DIFF_MODEL.pred_noise = args.pred_noise
    args.MODEL.DIFF_MODEL.time_steps = args.time_steps
    args.MODEL.DIFF_MODEL.denoise_steps = args.denoise_steps
    args.MODEL.DIFF_MODEL.scale = args.scale
    args.MODEL.DIFF_MODEL.scale_input = args.scale_input
    args.MODEL.DIFF_MODEL.noise_schedule = args.schedule
    args.MODEL.DIFF_MODEL.ddim = args.ddim
    args.MODEL.DIFF_MODEL.ddim_eta = args.ddim_eta
    
    transform = get_transform(args.DATA.input_resolution)
    input_resolution, output_resolution = args.DATA.input_resolution, args.DATA.output_resolution
    train_dataset = GazeFollow_weakly(args.DATA, transform, test=False, ratio = args.supervise_ratio, use_sup_only=False)
    val_dataset = GazeFollow_weakly(args.DATA, transform, test=True)
    
    labeled_idxs, unlabeled_idxs = get_label_idx(train_dataset)
    if args.labeled_batch_size>0:
        batch_sampler = TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)
    else:
        assert False, "labeled batch size {}".format(args.labeled_batch_size)
    
    if not args.debug:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_sampler = batch_sampler,
                                               num_workers=12)
    else:
        lst_train = random.choices(list(range(len(train_dataset))), k=4)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, 
            num_workers = 0, shuffle=False, sampler = torch.utils.data.SubsetRandomSampler(lst_train))
    
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=args.batch_size*2,
                                               shuffle=False,
                                               num_workers=12)
    if args.resume==-1:
        setting_name = f"{args.remark}model{args.model}_teacher{args.semi_model}_infsteps{args.inference_steps}_denoisesteps{args.denoise_steps}_ratio{args.supervise_ratio}_lr{args.lr}bs{args.batch_size}_onlyin{args.onlyin}_labelbs{args.labeled_batch_size}_ampfactor{args.loss_amp_factor}_lambda{args.lambda_}_weightdecay{args.weight_decay}_optim{args.optim}" 
        if args.debug:
            setting_name += '_debug'
    else:
        setting_name = args.setting_name
    
    logdir = os.path.join('./logs/semi_sup_gf', args.project_name, setting_name)
    ckpt_dir = os.path.join(args.ckpt_dir, args.project_name, setting_name)
    if args.resume==-1 and os.path.exists(logdir):
        shutil.rmtree(logdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists(ckpt_dir) and args.save_ckpt:
        os.makedirs(ckpt_dir)

    writer = SummaryWriter(logdir)
    np.random.seed(1)
    log_path = os.path.join(logdir, "{}.log".format('train'))
    logger = get_logger(log_path)
    logger.info(args)

    if args.model=='baseline':
        print('use baseline model!') 
        model = ModelSpatial(args.MODEL)
        
    eps = 1e-11   
    model.cuda()
    if args.semi_model=='diffusion':
        pseudo_model = GF_Diffusion_Model_new(args.MODEL)
    elif args.semi_model=='baseline_gradcam':
        pseudo_model = ModelSpatial_GradCAM(args.MODEL)
    elif args.semi_model=='baseline':
        pseudo_model = ModelSpatial(args.MODEL)
        
    myutils.load_pretrained_weights(pseudo_model, state_dict = None, weight_path=args.teacher_ckpt) 
    print("successfully load {} for {} pseudo model!".format(args.teacher_ckpt, args.semi_model))
    pseudo_model.cuda()
    pseudo_model.eval()
    if len(args.init_weights)>0:
        myutils.load_pretrained_weights(model, state_dict = None, weight_path=args.init_weights) 
       
    mseloss_mean = nn.MSELoss(reduction='mean')
    bcelogit_loss = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    
    if args.supervise_ratio!=0.5:
        sche = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15], gamma=0.2)
    else:
        sche = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25], gamma=0.2)
     
    step = 0
    if args.resume!=-1:
        step = myutils.resume_from_epoch(args, model, sche, ckpt_dir)   
    multigpu = False
    if len(args.device)>1:
        multigpu=True
        model = nn.DataParallel(model)
    
    loss_amp_factor = args.loss_amp_factor # multiplied to the loss to prevent underflow
    max_steps = len(train_loader)
    optimizer.zero_grad()
    print("Training in progress ...")
    total_gradcam_invalid = 0
    for ep in range(max(args.resume+1,0), args.epochs):
        logger.info(f"Epoch {ep}: learning rate: {optimizer.param_groups[0]['lr']}") 
        num_sup, num_all = 0,0
        
        for batch, (img, face, head_channel, gaze_heatmap, gaze_inside, head_coords, body_coords, gaze_coords, gradcam_resize, gaze_heatmap_maxgradcam, sup, gradcam_valid, path) in enumerate(train_loader):
            
            model.train(True)
            images, head, faces = img.cuda(), head_channel.cuda(), face.cuda()
            head_coords, gaze_coords = head_coords.float().cuda(), gaze_coords.float().cuda()
            gaze_heatmap = gaze_heatmap.cuda()
            gaze_heatmap_maxgradcam = gaze_heatmap_maxgradcam.cuda()
            gradcam_resize = gradcam_resize.cuda()
            sup, gradcam_valid = sup.cuda(), gradcam_valid.cuda()
            gaze_inside = gaze_inside.cuda().to(torch.float)
            
            
            gaze_heatmap_pred, inout_pred = model([images, head, faces])
            
            gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1)
            num_in = gaze_inside.sum().item()
            bs = img.size()[0]
            num_sup += sup.sum().item()
            num_all += sup.size()[0]
            sample_steps = args.denoise_steps if args.denoise_steps>0 else args.time_steps
            replace_idx = (1 - sup).bool()
            if not args.gradcam_only:
                with torch.no_grad():
                    
                    images, head, faces, gradcam_resize = images[replace_idx], head[replace_idx], faces[replace_idx], gradcam_resize[replace_idx]
                    t_exp = torch.tensor([sample_steps]).long().to(images.device)
                    hm_all = []
                     
                    if args.semi_model=='baseline_gradcam':
                        denoised_hm, _ = pseudo_model([images, head, faces, gradcam_resize])  # add gradcam to baseline model for prediction
                        
                    elif args.semi_model=='baseline':
                        denoised_hm, _ = pseudo_model([images, head, faces])  # baseline prediction, just use same name for convenience
                        
                    elif args.semi_model=='diffusion':
                        for i in range(args.inference_times):
                            if not args.sample_from_noise:
                                gradcam_noised, epsilon = pseudo_model.diffusion.q_sample(gradcam_resize, t_exp)                    
                                denoised_hm  = pseudo_model([images, head, faces], args.inference_steps, start_steps=args.denoise_steps, inter_sample=gradcam_noised)
                                hm_all.append(denoised_hm)
                            else:    
                                # ablation, directly sample from noise without gradcam, and use as pseudo heatmap
                                denoised_hm = pseudo_model([images, head, faces], args.inference_steps, start_steps=-1)  
                                hm_all.append(denoised_hm)
                    
                        denoised_hm = torch.mean(torch.stack(hm_all), dim=0)        
                    denoised_hm = denoised_hm.squeeze(1)
                    max_all, min_all = denoised_hm.flatten(start_dim=1).max(dim=1)[0].unsqueeze(-1).unsqueeze(-1), denoised_hm.flatten(start_dim=1).min(dim=1)[0].unsqueeze(-1).unsqueeze(-1)
                    denoised_hm = (denoised_hm - min_all) / (max_all - min_all + eps)  # norm to [0,1] 
                    gaze_heatmap[replace_idx] = denoised_hm
                    
                    # l2 loss computed only for inside case?
                    if args.semi_model!='baseline':
                        replace_idx = torch.logical_and(replace_idx, gradcam_valid)
                    
            else:
                gaze_heatmap[replace_idx] = gradcam_resize[replace_idx].squeeze(1)
            
            loss_apply_idx = torch.logical_or(replace_idx, sup.bool())
            
            sup_mask = sup.bool()
            sup_inside = torch.logical_and(gaze_inside.bool(), sup_mask)
            unsup_inside = torch.logical_and(gaze_inside.bool(), ~sup_mask)
            inside_mask = torch.logical_or(sup_inside, unsup_inside).bool()
            loss_apply_idx = torch.logical_and(loss_apply_idx, inside_mask) 
            
            
            l2_loss = l2_hm_loss(gaze_heatmap_pred, gaze_heatmap, loss_apply_idx, onlyin=args.onlyin)
            l2_loss = l2_loss * loss_amp_factor
            
            # cross entropy loss for in vs out
            if args.no_inout:
                loss_inout = torch.tensor(0.0).cuda()
            else:
                loss_inout = bcelogit_loss(inout_pred.squeeze(), gaze_inside.squeeze())
            
            # sampling
            Xent_loss = loss_inout  * args.lambda_
            if torch.isnan(Xent_loss):
                pdb.set_trace()
            
            total_loss = l2_loss + Xent_loss 
            invalid_idx = torch.logical_and((1-sup).bool(), ~gradcam_valid)
            num_invalid = torch.logical_and(invalid_idx, gaze_inside).sum()
            total_gradcam_invalid += num_invalid
            
            writer.add_scalar('train/Loss_step', total_loss.item(), step)
            writer.add_scalar('train/l2_loss_step', l2_loss.item(), step)
            writer.add_scalar('train/Inout_loss_step', Xent_loss.item(), step)
            

            total_loss.backward() 
            optimizer.step()
            optimizer.zero_grad()

            step += 1
            if batch % args.print_every == 0:
                print("Epoch:{:04d}\tstep:{:06d}/{:06d}\ttraining loss: (l2){:.4f} (Inout){:.4f}".format(ep, batch+1, max_steps, l2_loss, Xent_loss))
                sup_ratio = num_sup / num_all
                print("num sup: {}, num unsup: {}, num valid unsup: {}, supervise_ratio:{:.2f}%".format(sup.sum(), bs - sup.sum(), torch.sum(replace_idx), sup_ratio*100))
                writer.add_scalar("Train Loss", total_loss, global_step=step)
        
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], ep)
        sup_ratio = num_sup / num_all
        
        if not args.debug and ep % args.eval_every==0:
            logger.info('Validation in progress ...: epoch {}'.format(ep))
            model.train(False)
            AUC = []; min_dist = []; avg_dist = []
            ep_val_loss, ep_val_l2_loss, ep_val_inout_loss = 0.0, 0.0, 0.0
            num_imgs = len(val_dataset)
            start_idx, end_idx = 0,0
            with torch.no_grad():
                for val_batch, (img, face, head_channel, gaze_heatmap, cont_gaze, imsize, head_coords, body_coords, gaze_coords, gradcam_resize, gradcam_valid, path) in enumerate(val_loader):
                    start_idx, end_idx = end_idx, start_idx + img.size()[0]
                    images, head, faces = img.cuda(), head_channel.cuda(), face.cuda()
                    head_coords, gaze_coords = head_coords.float().cuda(), gaze_coords.float().cuda()
                    gaze_heatmap = gaze_heatmap.cuda()
                
                    gaze_heatmap_pred, inout_pred = model([images, head, faces])
                    gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1)
                    bs = img.size()[0]
                    
                    # l2 loss computed only for inside case
                    l2_loss = mseloss_mean(gaze_heatmap_pred, gaze_heatmap)*loss_amp_factor
                    gaze_inside = torch.ones(images.size()[0]).cuda()
                    
                    max_all, min_all = gaze_heatmap_pred.flatten(start_dim=1).max(dim=1)[0].unsqueeze(-1).unsqueeze(-1), gaze_heatmap_pred.flatten(start_dim=1).min(dim=1)[0].unsqueeze(-1).unsqueeze(-1)
                    gaze_heatmap_pred = (gaze_heatmap_pred - min_all) / (max_all - min_all + eps)
                             
                    if args.no_inout:
                        loss_inout = torch.tensor(0.0).cuda()
                    else:
                        loss_inout = bcelogit_loss(inout_pred.squeeze(), gaze_inside.squeeze())
            
                    Xent_loss = loss_inout  * args.lambda_
                    total_loss = l2_loss + Xent_loss

                    ep_val_l2_loss += l2_loss.item()
                    ep_val_inout_loss += Xent_loss.item()
                    ep_val_loss += total_loss.item()
                    gaze_heatmap_pred = gaze_heatmap_pred.cpu().numpy()
                    
            
                    for b_i in range(len(cont_gaze)):
                        # remove padding and recover valid ground truth points
                        valid_gaze = cont_gaze[b_i]
                        valid_gaze = valid_gaze[valid_gaze != -1].view(-1,2)
                        # AUC: area under curve of ROC
                        multi_hot = imutils.multi_hot_targets(cont_gaze[b_i], imsize[b_i])
                        scaled_heatmap = resize(gaze_heatmap_pred[b_i], (imsize[b_i][1], imsize[b_i][0]))
                        auc_score = evaluation.auc(scaled_heatmap, multi_hot)
                        AUC.append(auc_score)
                        # min distance: minimum among all possible pairs of <ground truth point, predicted point>
                        pred_x, pred_y = evaluation.argmax_pts(gaze_heatmap_pred[b_i])
                        norm_p = [pred_x/float(output_resolution), pred_y/float(output_resolution)]
                        all_distances = []
                        valid_gaze = valid_gaze.numpy()
                        for gt_gaze in valid_gaze:
                            all_distances.append(evaluation.L2_dist(gt_gaze, norm_p))
                        min_dist.append(min(all_distances))
                        # average distance: distance between the predicted point and human average point
                        mean_gt_gaze = np.mean(valid_gaze, axis=0)
                        avg_distance = evaluation.L2_dist(mean_gt_gaze, norm_p)
                        avg_dist.append(avg_distance)
                
               
                logger.info("\tAUC:{:.4f}\tmin dist:{:.4f}\tavg dist:{:.4f}\t ".format(
                    torch.mean(torch.tensor(AUC)),
                    torch.mean(torch.tensor(min_dist)),
                    torch.mean(torch.tensor(avg_dist))))
            writer.add_scalar('Validation/AUC', torch.mean(torch.tensor(AUC)), global_step=ep)
            writer.add_scalar('Validation/min dist', torch.mean(torch.tensor(min_dist)), global_step=ep)
            writer.add_scalar('Validation/avg dist', torch.mean(torch.tensor(avg_dist)), global_step=ep)
            ep_val_loss, ep_val_l2_loss, ep_val_inout_loss = ep_val_loss/(val_batch+1), ep_val_l2_loss/(val_batch+1), ep_val_inout_loss/(val_batch+1)
            writer.add_scalar('val/Loss', ep_val_loss, ep)
            writer.add_scalar('val/l2_loss', ep_val_l2_loss, ep)
            writer.add_scalar('val/Inout_loss', ep_val_inout_loss, ep)
            logger.info("Epoch {} test: Total loss: {} L2 loss: {}, inout loss: {}".format(ep, ep_val_loss, ep_val_l2_loss, ep_val_inout_loss))
        
            if ep % args.save_every == 0 and args.save_ckpt:
                # save the model
                if multigpu:
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                checkpoint = {'state_dict': state_dict, 'lr':optimizer.param_groups[0]['lr'], 'train_step':step}
                save_checkpoint(checkpoint, ckpt_dir, 'epoch_%02d_weights.pt' % (ep), remove_module_from_keys=True)

        sche.step() 
        writer.flush()
    writer.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='0', help="gpu id")
    parser.add_argument('--project_name', default='gf_semi_diffusion')
    parser.add_argument('--setting_name', default='')
    parser.add_argument('--onlyin', action='store_true')
    parser.add_argument('--config', default='./config/config_diffusion.yaml')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--no_inout', action='store_true')
    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--optim', default='adam')
    parser.add_argument('--model', default='baseline')
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--loss_amp_factor', type=float, default=10000)
    parser.add_argument('--lambda_', type=float, default=20.0)
    parser.add_argument("--lr", type=float, default=2.5e-4, help="learning rate")
    parser.add_argument('--supervise_ratio', type=float, default=1.0, help='ratio of data with ground truth annotation in semi-supervised training')
    parser.add_argument("--batch_size", type=int, default=80, help="batch size")
    parser.add_argument("--labeled_batch_size", type=int, default=-1, help="batch size of supervised samples")
    parser.add_argument("--epochs", type=int, default=25, help="number of epochs")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay")
    parser.add_argument("--print_every", type=int, default=100, help="print every ___ iterations")
    parser.add_argument("--save_every", type=int, default=1, help="save every ___ epochs")
    parser.add_argument("--resume", type=int, default=-1, help="which epoch to resume from")
    parser.add_argument('--remark', type=str, default='')
    parser.add_argument('--semi_model', default='diffusion')
    parser.add_argument('--pred_noise', action='store_true')
    parser.add_argument('--scale', type=float, default=2.0)
    parser.add_argument('--scale_input', action='store_true')
    parser.add_argument('--ddim_eta', type=float, default=1.0)
    parser.add_argument('--denoise_steps', type=int, default=250)
    parser.add_argument('--sample_from_noise', action='store_true')
    parser.add_argument('--no_ddim', dest='ddim', action='store_false')
    parser.add_argument('--schedule', type=str, default='linear')
    parser.add_argument('--gradcam_only', action='store_true')
    parser.add_argument('--inference_times', type=int, default=1)
    parser.add_argument('--inference_steps', type=int, default=2, help='average the sampled outputs during inference as the final output')
    parser.add_argument('--time_steps', type=int, default=500, help='steps in diffusion training')
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--init_weights", type=str, default="/nfs/bigcortex.cs.stonybrook.edu/add_disk0/qiaomu/ckpts/gaze/videoatttarget/initial_weights_for_spatial_training.pt", help="initial weights")
    parser.add_argument("--ckpt_dir", type=str, default="/nfs/bigrod/add_disk0/qiaomu/ckpts/gazefollow_semisup", help="directory to save log files")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    train(args)
    
    