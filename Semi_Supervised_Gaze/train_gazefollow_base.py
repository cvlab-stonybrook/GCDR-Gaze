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
from utils.torchtools import save_checkpoint
from model.model_weakly import ModelSpatial, ModelSpatial_GradCAM
eps = 1e-8



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
    args.MODEL.use_dir = False
    args.DATA.gradcam_outsize = args.gradcam_outsize
    
    transform = get_transform(args.DATA.input_resolution)
    input_resolution, output_resolution = args.DATA.input_resolution, args.DATA.output_resolution
    train_dataset = GazeFollow_weakly(args.DATA, transform, test=False, ratio = args.supervise_ratio, use_sup_only=True)
    val_dataset = GazeFollow_weakly(args.DATA, transform, test=True)
    
    if not args.debug:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=16)
    else:
        lst_train = random.choices(list(range(len(train_dataset))), k=4)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, 
            num_workers = 0, shuffle=False, sampler = torch.utils.data.SubsetRandomSampler(lst_train))
    
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=args.batch_size*2,
                                               shuffle=False,
                                               num_workers=16)
    if args.resume==-1:
        setting_name = f"{args.remark}model{args.model}_ratio{args.supervise_ratio}_lr{args.lr}bs{args.batch_size}_onlyin{args.onlyin}_ampfactor{args.loss_amp_factor}_lambda{args.lambda_}_weightdecay{args.weight_decay}_optim{args.optim}" 
        if args.debug:
            setting_name += '_debug'
    else:
        setting_name = args.setting_name
    
    logdir = os.path.join('./logs/semi_sup_gf', args.project_name, setting_name)
    ckpt_dir = os.path.join(args.ckpt_dir, 'semi_sup_gf', args.project_name, setting_name)
    if args.resume==-1 and os.path.exists(logdir):
        shutil.rmtree(logdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    writer = SummaryWriter(logdir)
    np.random.seed(1)
    log_path = os.path.join(logdir, "{}.log".format('train'))
    logger = get_logger(log_path)
    logger.info(args)

    if args.model=='baseline':
        model = ModelSpatial(args.MODEL)
    elif args.model=='baseline_gradcam':
        model = ModelSpatial_GradCAM(args.MODEL)
    model.cuda()
    
    if len(args.init_weights)>0 and 'imagenet' not in args.model:
        myutils.load_pretrained_weights(model, state_dict = None, weight_path=args.init_weights) 
        
    mse_loss = nn.MSELoss(reduction='none') # not reducing in order to ignore outside cases
    mseloss_mean = nn.MSELoss(reduction='mean')
    bcelogit_loss = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if not args.no_decay:
        
        sche = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 40], gamma=0.2)
    
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

    for ep in range(max(args.resume+1,0), args.epochs):
        logger.info(f"Epoch {ep}: learning rate: {optimizer.param_groups[0]['lr']}") 
        num_sup, num_all = 0,0
        
        for batch, (img, face, head_channel, gaze_heatmap, gaze_inside, head_coords, body_coords, gaze_coords, gradcam_resize, hm_maxgradcam, sup, gradcam_valid, path) in enumerate(train_loader):
            
            model.train(True)
            images, head, faces = img.cuda(), head_channel.cuda(), face.cuda()
            head_coords, gaze_coords = head_coords.float().cuda(), gaze_coords.float().cuda()
            gaze_heatmap = gaze_heatmap.cuda()
            gradcam_resize = gradcam_resize.cuda()
            gradcam_valid = gradcam_valid.cuda().bool()
            gaze_inside = gaze_inside.cuda().to(torch.float)
            if args.model=='baseline_gradcam':
                gaze_heatmap_pred, inout_pred = model([images, head, faces, gradcam_resize])
            else:
                gaze_heatmap_pred, inout_pred = model([images, head, faces])
            gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1)
            num_in = gaze_inside.sum().item()
            num_out = img.size()[0]-num_in
            bs = img.size()[0]
            num_sup += sup.sum().item()
            num_all += sup.size()[0]
            sup_mask = sup.cuda().bool()
            if args.model=='baseline_gradcam':
                sup_mask = torch.logical_and(sup_mask, gradcam_valid)

                # l2 loss computed only for inside case?
            if not args.onlyin:
                l2_loss = mseloss_mean(gaze_heatmap_pred, gaze_heatmap)
            else:
                l2_loss = mse_loss(gaze_heatmap_pred, gaze_heatmap) # only for supervised labels
                l2_loss = torch.mean(l2_loss, dim=1)
                l2_loss = torch.mean(l2_loss, dim=1)
                l2_loss = torch.mul(l2_loss, gaze_inside)
                l2_loss = torch.sum(l2_loss)/gaze_inside.sum()


            l2_loss = l2_loss * loss_amp_factor
            
            if args.no_inout:
                loss_inout = torch.tensor(0.0).cuda()
            else:
                loss_inout = bcelogit_loss(inout_pred.squeeze(), gaze_inside.squeeze())
            
            # sampling

            Xent_loss = loss_inout  * args.lambda_
            
            total_loss = l2_loss + Xent_loss 
            if torch.isnan(total_loss):
                pdb.set_trace()
            
            
            total_loss.backward() 
            optimizer.step()
            optimizer.zero_grad()

            step += 1

            if batch % args.print_every == 0:
                print("Epoch:{:04d}\tstep:{:06d}/{:06d}\ttraining loss: (l2){:.4f} (Inout){:.4f}".format(ep, batch+1, max_steps, l2_loss, Xent_loss))
                # Tensorboard
        
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], ep)
        
        if not args.debug and ep % args.eval_every==0:
            logger.info('Validation in progress ...: epoch {}'.format(ep))
            model.train(False)
            AUC = []; min_dist = []; avg_dist = []
            ep_val_loss, ep_val_l2_loss, ep_val_inout_loss = 0.0, 0.0, 0.0
            num_imgs = len(val_dataset)
            with torch.no_grad():
                for val_batch, (img, face, head_channel, gaze_heatmap,  cont_gaze, imsize, head_coords, body_coords, gaze_coords, gradcam_resize, valid, path) in enumerate(val_loader):
                    images, head, faces = img.cuda(), head_channel.cuda(), face.cuda()
                    head_coords, gaze_coords = head_coords.float().cuda(), gaze_coords.float().cuda()
                    gaze_heatmap = gaze_heatmap.cuda()
                    gradcam_resize = gradcam_resize.cuda()
                    if args.model=='baseline_gradcam':
                        gaze_heatmap_pred, inout_pred = model([images, head, faces, gradcam_resize])
                    else:
                        gaze_heatmap_pred, inout_pred = model([images, head, faces])
                    gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1)
                    
                    # Loss
                        # l2 loss computed only for inside case
                    l2_loss = mseloss_mean(gaze_heatmap_pred, gaze_heatmap)*loss_amp_factor
                    gaze_inside = torch.ones(images.size()[0]).cuda()
                    
                    loss_inout = torch.tensor(0.0).cuda()
                    total_loss = l2_loss + loss_inout

                    ep_val_l2_loss += l2_loss.item()
                    ep_val_inout_loss += loss_inout.item()
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
                
                logger.info("\tAvg dist:{:.4f}\tMin dist:{:.4f}\tAUC:{:.4f}\t".format(
                    torch.mean(torch.tensor(avg_dist)),
                    torch.mean(torch.tensor(min_dist)),
                    torch.mean(torch.tensor(AUC))
                    ))
                
            writer.add_scalar('Validation AUC', torch.mean(torch.tensor(AUC)), global_step=ep)
            writer.add_scalar('Validation min dist', torch.mean(torch.tensor(min_dist)), global_step=ep)
            writer.add_scalar('Validation avg dist', torch.mean(torch.tensor(avg_dist)), global_step=ep)
            ep_val_loss, ep_val_l2_loss, ep_val_inout_loss = ep_val_loss/(val_batch+1), ep_val_l2_loss/(val_batch+1), ep_val_inout_loss/(val_batch+1)
            writer.add_scalar('val/Loss', ep_val_loss, ep)
            writer.add_scalar('val/l2_loss', ep_val_l2_loss, ep)
            writer.add_scalar('val/Inout_loss', ep_val_inout_loss, ep)

            logger.info("Epoch {} test: Total loss: {} L2 loss: {}, inout loss: {}".format(ep, ep_val_loss, ep_val_l2_loss, ep_val_inout_loss))

            
            if ep % args.save_every == 0 and not args.debug:
                # save the model
                if multigpu:
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                checkpoint = {'state_dict': state_dict, 'lr':optimizer.param_groups[0]['lr'], 'train_step':step}
                save_checkpoint(checkpoint, ckpt_dir, 'epoch_%02d_weights.pt' % (ep), remove_module_from_keys=True)

        if not args.no_decay:
            sche.step() 
        writer.flush()
    writer.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='0', help="gpu id")
    parser.add_argument('--project_name', default='gazefollow_semi')
    parser.add_argument('--setting_name', default='')
    parser.add_argument('--onlyin', action='store_true')
    parser.add_argument('--config', default='./config/config_diffusion.yaml')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--gradcam_outsize', type=int, default=64)
    parser.add_argument('--no_inout', action='store_true')
    parser.add_argument('--optim', default='adam')
    parser.add_argument('--model', default='baseline_single')
    parser.add_argument('--plot_every', type=int, default=400)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--loss_amp_factor', type=float, default=10000)
    parser.add_argument('--lambda_', type=float, default=20.0)
    parser.add_argument("--lr", type=float, default=2.5e-4, help="learning rate")
    parser.add_argument('--no_decay', action='store_true')
    parser.add_argument('--supervise_ratio', type=float, default=1.0, help='ratio of data with ground truth annotation in semi-supervised training')
    parser.add_argument("--batch_size", type=int, default=48, help="batch size")
    parser.add_argument("--epochs", type=int, default=40, help="number of epochs")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay")
    parser.add_argument("--print_every", type=int, default=100, help="print every ___ iterations")
    parser.add_argument("--save_every", type=int, default=1, help="save every ___ epochs")
    parser.add_argument("--resume", type=int, default=-1, help="which epoch to resume from")
    parser.add_argument('--remark', type=str, default='')
    parser.add_argument("--init_weights", type=str, default="/nfs/bigcortex/add_disk0/qiaomu/ckpts/gaze/videoatttarget/initial_weights_for_spatial_training.pt", help="initial weights")
    parser.add_argument("--ckpt_dir", type=str, default="/nfs/bigrod/add_disk0/qiaomu/ckpts/gaze/gazefollow_semisup", help="directory to save log files")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    
    train(args)