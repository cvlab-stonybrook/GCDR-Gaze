import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from model.model_base import Bottleneck, BottleneckConvLSTM
from diffusion.unet import UNetModel
from torch.nn.utils.rnn import PackedSequence
from diffusion.gaussian_diffusion_new import GaussianDiffusion
from lib.pytorch_convolutional_rnn_new import convolutional_rnn as convolutional_rnn_new

class Gazefollow_Encoder(nn.Module):
    
    def __init__(self):
        block = Bottleneck
        layers_scene = [3, 4, 6, 3, 2]
        layers_face = [3, 4, 6, 3, 2]
        self.inplanes_scene = 64
        self.inplanes_face = 64
        super(Gazefollow_Encoder, self).__init__()
        # common
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        # scene pathway
        self.conv1_scene = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_scene = nn.BatchNorm2d(64)
        self.layer1_scene = self._make_layer_scene(block, 64, layers_scene[0])
        self.layer2_scene = self._make_layer_scene(block, 128, layers_scene[1], stride=2)
        self.layer3_scene = self._make_layer_scene(block, 256, layers_scene[2], stride=2)
        self.layer4_scene = self._make_layer_scene(block, 512, layers_scene[3], stride=2)
        self.layer5_scene = self._make_layer_scene(block, 256, layers_scene[4], stride=1) # additional to resnet50

        # face pathway
        self.conv1_face = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1_face = nn.BatchNorm2d(64)
        self.layer1_face = self._make_layer_face(block, 64, layers_face[0])
        self.layer2_face = self._make_layer_face(block, 128, layers_face[1], stride=2)
        self.layer3_face = self._make_layer_face(block, 256, layers_face[2], stride=2)
        self.layer4_face = self._make_layer_face(block, 512, layers_face[3], stride=2)
        self.layer5_face = self._make_layer_face(block, 256, layers_face[4], stride=1) # additional to resnet50

        # attention
        self.attn = nn.Linear(1808, 1*7*7)


    def _make_layer_scene(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_scene != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes_scene, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_scene, planes, stride, downsample))
        self.inplanes_scene = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_scene, planes))

        return nn.Sequential(*layers)
    
    
    def _make_layer_face(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_face != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes_face, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_face, planes, stride, downsample))
        self.inplanes_face = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_face, planes))

        return nn.Sequential(*layers)
    
    def forward(self, input):
        images, head, face = input
        
        face = self.conv1_face(face)
        face = self.bn1_face(face)
        face = self.relu(face)
        face = self.maxpool(face)
        face = self.layer1_face(face)
        face = self.layer2_face(face)
        face = self.layer3_face(face)
        face = self.layer4_face(face)
        face_feat = self.layer5_face(face)

        # reduce head channel size by max pooling: (N, 1, 224, 224) -> (N, 1, 28, 28)
        head_reduced = self.maxpool(self.maxpool(self.maxpool(head))).view(-1, 784)
        # reduce face feature size by avg pooling: (N, 1024, 7, 7) -> (N, 1024, 1, 1)
        face_feat_reduced = self.avgpool(face_feat).view(-1, 1024)
        # get and reshape attention weights such that it can be multiplied with scene feature map
        
        attn_weights = self.attn(torch.cat((head_reduced, face_feat_reduced), 1))

        attn_weights = attn_weights.view(-1, 1, 49)
        attn_weights = F.softmax(attn_weights, dim=2) # soft attention weights single-channel
        attn_weights = attn_weights.view(-1, 1, 7, 7)

        im = torch.cat((images, head), dim=1)
        im = self.conv1_scene(im)
        im = self.bn1_scene(im)
        im = self.relu(im)
        im = self.maxpool(im)
        im = self.layer1_scene(im)
        im = self.layer2_scene(im)
        im = self.layer3_scene(im)
        im = self.layer4_scene(im)
        scene_feat = self.layer5_scene(im)
        # attn_weights = torch.ones(attn_weights.shape)/49.0
        attn_applied_scene_feat = torch.mul(attn_weights, scene_feat) # (N, 1, 7, 7) # applying attention weights on scene feat

        scene_face_feat = torch.cat((attn_applied_scene_feat, face_feat), 1)
        
        return scene_face_feat
 

class GF_Diffusion_Model_new(nn.Module):
    def __init__(self, args):
        super(GF_Diffusion_Model_new, self).__init__()
        self.feat_extractor = Gazefollow_Encoder()
        self.relu = nn.ReLU(inplace=True)
        args_diff = args.DIFF_MODEL
        self.diffusion_model = UNetModel(image_size=args_diff.img_size, in_channels=args_diff.in_channels, out_channels=args_diff.out_channels, 
                model_channels=args_diff.model_channels, num_res_blocks=args_diff.num_res_blocks, channel_mult=args_diff.channel_mult,
                attention_resolutions=args_diff.attention_resolutions, num_heads=args_diff.num_heads, use_scale_shift_norm=False)
        self.maxpool = nn.MaxPool2d(7, stride=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        
        out_channels = args_diff.model_channels * args_diff.channel_mult[3]  # 256 here
        self.compress_conv1 = nn.Conv2d(2048, out_channels, kernel_size=1, stride=1, padding=0, bias=False) # ori: 2048, with att: 512
        self.compress_bn1 = nn.BatchNorm2d(out_channels)
        
        up_channel_2 = args_diff.model_channels * args_diff.channel_mult[2]
        up_channel_1 = args_diff.model_channels * args_diff.channel_mult[1]
        up_channel_0 = args_diff.model_channels * args_diff.channel_mult[0]
        self.upsample_3 = nn.Upsample((8,8))
        self.upsample_2 = nn.ConvTranspose2d(256, up_channel_2, kernel_size=2, stride=2)  # outsize: 16x16
        self.upsample2_bn = nn.BatchNorm2d(up_channel_2)
        self.upsample_1 = nn.ConvTranspose2d(up_channel_2, up_channel_1, kernel_size=2, stride=2)  # outsize 32x32
        self.upsample1_bn = nn.BatchNorm2d(up_channel_1)
        self.upsample_0 = nn.ConvTranspose2d(up_channel_1, up_channel_0, kernel_size=2, stride=2)  # outsize 64x64
        self.upsample0_bn = nn.BatchNorm2d(up_channel_0)
        
        
        self.num_steps = args_diff.time_steps
        self.clip_denoised = args_diff.clip_denoised
        self.inference_steps = args_diff.inference_steps
        self.pred_noise = args_diff.pred_noise
        self.scale_input = args_diff.scale_input
        self.scale = args_diff.scale
        self.ddim = args_diff.ddim
        self.ddim_eta = args_diff.ddim_eta
        self.diffusion = GaussianDiffusion(steps=self.num_steps, denoise_steps=-1, schedule=args_diff.noise_schedule, scale=args_diff.scale, pred_noise=args_diff.pred_noise, scale_input=args_diff.scale_input, ddim_eta=self.ddim_eta)
        self.hm_size =  (1, 64, 64)   
    
    def forward(self, input, inference_steps, start_steps=-1, t=None, inter_sample=None, annt_gen=None):
        if self.training:
            return self.forward_train(input, t=t)
        else:
            return self.forward_inference(input, inference_steps, start_steps=start_steps, inter_sample=inter_sample, annt_gen=annt_gen)
    
    
    def forward_train(self, input, t=None):
        
        images, head, face, gt_hm = input
        feature = self.feat_extractor([images, head, face])
        feature = self.compress_conv1(feature)
        feature = self.compress_bn1(feature)
        feature = self.relu(feature)
        
        context_layer3 = self.upsample_3(feature)
        context_layer2 = self.upsample_2(context_layer3)
        context_layer2 = self.upsample2_bn(context_layer2)
        context_layer2 = self.relu(context_layer2)
        context_layer1 = self.upsample_1(context_layer2)
        context_layer1 = self.upsample1_bn(context_layer1)
        context_layer1 = self.relu(context_layer1)
        context_layer0 = self.upsample_0(context_layer1)
        context_layer0 = self.upsample0_bn(context_layer0)
        context_layer0 = self.relu(context_layer0)
        
        if t is None:
            t = torch.randint(1, self.diffusion.num_timesteps+1, (images.shape[0],)).long().to(images.device)
        x0 = gt_hm.clone()
        
        hm_t, epsilon = self.diffusion.q_sample(x0, t-1)
    
        t = t.float()
        eps_est = self.diffusion_model(hm_t, t, context_list=[context_layer0, context_layer1, context_layer2, context_layer3])
        
        t = t.long() 
        return eps_est, epsilon, t 

    def forward_inference(self, input, inference_steps, start_steps=-1, inter_sample=None, annt_gen=None):
        images, head, face = input
        feature = self.feat_extractor([images, head, face])
        feature = self.compress_conv1(feature)
        feature = self.compress_bn1(feature)
        feature = self.relu(feature)
        context_layer3 = self.upsample_3(feature)
        context_layer2 = self.upsample_2(context_layer3)
        context_layer2 = self.upsample2_bn(context_layer2)
        context_layer2 = self.relu(context_layer2)
        context_layer1 = self.upsample_1(context_layer2)
        context_layer1 = self.upsample1_bn(context_layer1)
        context_layer1 = self.relu(context_layer1)
        context_layer0 = self.upsample_0(context_layer1)
        context_layer0 = self.upsample0_bn(context_layer0)
        context_layer0 = self.relu(context_layer0)
        
        bs = images.size(0)
        
        if self.ddim:
        
            inference_out = self.diffusion.ddim_sample_new(self.diffusion_model, inference_steps, start_steps, shape=(bs,) + self.hm_size, context=[context_layer0, context_layer1, context_layer2, context_layer3], 
                                                                         eta=self.ddim_eta, x_T=inter_sample, annt_gen=annt_gen)
        
            
        return inference_out
    
    
    def denoise_with_unsup(self, input, sup_mask, t=None):
        images, head, face, gt_hm = input 
        bs = images.size(0)
        shape = (bs,) + self.hm_size
        noise_input = torch.randn(shape, device=images.device)
        
        if t is None:
            t = torch.randint(1, self.diffusion.num_timesteps+1, (images.shape[0],)).long().to(images.device)
        x0 = gt_hm.clone()
        hm_t, epsilon = self.diffusion.q_sample(x0, t-1)
        # replace the unsupervised hm_t
        hm_t[~sup_mask] = noise_input[~sup_mask]
        t[~sup_mask] = self.diffusion.num_timesteps
        t = t.float()
        
        feature = self.feat_extractor([images, head, face])
        feature = self.compress_conv1(feature)
        feature = self.compress_bn1(feature)
        feature = self.relu(feature)
        context_layer3 = self.upsample_3(feature)
        context_layer2 = self.upsample_2(context_layer3)
        context_layer2 = self.upsample2_bn(context_layer2)
        context_layer2 = self.relu(context_layer2)
        context_layer1 = self.upsample_1(context_layer2)
        context_layer1 = self.upsample1_bn(context_layer1)
        context_layer1 = self.relu(context_layer1)
        context_layer0 = self.upsample_0(context_layer1)
        context_layer0 = self.upsample0_bn(context_layer0)
        context_layer0 = self.relu(context_layer0)
        
        eps_est = self.diffusion_model(hm_t, t, context_list=[context_layer0, context_layer1, context_layer2, context_layer3])
        t = t.long()
        return eps_est, epsilon, t
    
      

class GF_Diffusion_Model_temporal(nn.Module):
    def __init__(self, args):
        super(GF_Diffusion_Model_temporal, self).__init__()
        self.feat_extractor = Gazefollow_Encoder()
        self.relu = nn.ReLU(inplace=True)
        args_diff = args.DIFF_MODEL
        self.diffusion_model = UNetModel(image_size=args_diff.img_size, in_channels=args_diff.in_channels, out_channels=args_diff.out_channels, 
                model_channels=args_diff.model_channels, num_res_blocks=args_diff.num_res_blocks, channel_mult=args_diff.channel_mult,
                attention_resolutions=args_diff.attention_resolutions, num_heads=args_diff.num_heads, use_scale_shift_norm=False)
        self.maxpool = nn.MaxPool2d(7, stride=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        
        out_channels = args_diff.model_channels * args_diff.channel_mult[3]  # 256 here
        self.compress_conv1 = nn.Conv2d(2048, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1 = nn.BatchNorm2d(out_channels)
        
        up_channel_2 = args_diff.model_channels * args_diff.channel_mult[2]
        up_channel_1 = args_diff.model_channels * args_diff.channel_mult[1]
        up_channel_0 = args_diff.model_channels * args_diff.channel_mult[0]
        self.upsample_3 = nn.Upsample((8,8))
        self.upsample_2 = nn.ConvTranspose2d(256, up_channel_2, kernel_size=2, stride=2)  # outsize: 16x16
        self.upsample2_bn = nn.BatchNorm2d(up_channel_2)
        self.upsample_1 = nn.ConvTranspose2d(up_channel_2, up_channel_1, kernel_size=2, stride=2)  # outsize 32x32
        self.upsample1_bn = nn.BatchNorm2d(up_channel_1)
        self.upsample_0 = nn.ConvTranspose2d(up_channel_1, up_channel_0, kernel_size=2, stride=2)  # outsize 64x64
        self.upsample0_bn = nn.BatchNorm2d(up_channel_0)
        

        self.num_steps = args_diff.time_steps
        self.clip_denoised = args_diff.clip_denoised
        self.inference_steps = args_diff.inference_steps
        self.pred_noise = args_diff.pred_noise
        self.scale_input = args_diff.scale_input
        self.scale = args_diff.scale
        self.ddim = args_diff.ddim
        self.ddim_eta = args_diff.ddim_eta
        self.diffusion = GaussianDiffusion(steps=self.num_steps, denoise_steps=-1, schedule=args_diff.noise_schedule, scale=args_diff.scale, pred_noise=args_diff.pred_noise, scale_input=args_diff.scale_input, ddim_eta=self.ddim_eta, annt_weight_temp=1.0)
        self.hm_size =  (1, 64, 64)   
        num_lstm_layers = 2
        
        self.convlstm_scene = convolutional_rnn_new.Conv2dLSTM(in_channels=256,
                                                     out_channels=256,
                                                     kernel_size=3,
                                                     num_layers=num_lstm_layers,
                                                     bidirectional=False,
                                                     batch_first=True,
                                                     stride=1,
                                                     dropout=0.5)
    
    def forward(self, input, inference_steps, start_steps=-1, inter_sample=None, hidden_feat: tuple = None, batch_sizes: list = None):
        if self.training:
            return self.forward_train(input, hidden_feat=hidden_feat, batch_sizes=batch_sizes)
        else:
            return self.forward_inference(input, inference_steps, start_steps=start_steps, inter_sample=inter_sample, hidden_feat=hidden_feat, batch_sizes=batch_sizes)
    
    
    def forward_train(self, input, hidden_feat: tuple = None, batch_sizes: list = None):
        
        images, head, face, gt_hm = input
        feature = self.feat_extractor([images, head, face])
        feature = self.compress_conv1(feature)
        feature = self.compress_bn1(feature)
        feature = self.relu(feature)
        
        x_pad = PackedSequence(feature, batch_sizes)
        y, hx = self.convlstm_scene(x_pad, hx=hidden_feat)
        feature = y.data
        
        context_layer3 = self.upsample_3(feature)
        context_layer2 = self.upsample_2(context_layer3)
        context_layer2 = self.upsample2_bn(context_layer2)
        context_layer2 = self.relu(context_layer2)
        context_layer1 = self.upsample_1(context_layer2)
        context_layer1 = self.upsample1_bn(context_layer1)
        context_layer1 = self.relu(context_layer1)
        context_layer0 = self.upsample_0(context_layer1)
        context_layer0 = self.upsample0_bn(context_layer0)
        context_layer0 = self.relu(context_layer0)
        
        t = torch.randint(1, self.diffusion.num_timesteps+1, (images.shape[0],)).long().to(images.device)
        
        x0 = gt_hm.clone()
        
        hm_t, epsilon = self.diffusion.q_sample(x0, t-1)
    
        t = t.float()
        eps_est = self.diffusion_model(hm_t, t, context_list=[context_layer0, context_layer1, context_layer2, context_layer3])
        
        return eps_est, epsilon, hx 
    
    def forward_inference(self, input, inference_steps, start_steps=-1, inter_sample=None, annt_gen=None, hidden_feat: tuple = None, batch_sizes: list = None):
        images, head, face = input
        feature = self.feat_extractor([images, head, face])
        feature = self.compress_conv1(feature)
        feature = self.compress_bn1(feature)
        feature = self.relu(feature)
        
        x_pad = PackedSequence(feature, batch_sizes)
        y, hx = self.convlstm_scene(x_pad, hx=hidden_feat)
        feature = y.data
        
        context_layer3 = self.upsample_3(feature)
        context_layer2 = self.upsample_2(context_layer3)
        context_layer2 = self.upsample2_bn(context_layer2)
        context_layer2 = self.relu(context_layer2)
        context_layer1 = self.upsample_1(context_layer2)
        context_layer1 = self.upsample1_bn(context_layer1)
        context_layer1 = self.relu(context_layer1)
        context_layer0 = self.upsample_0(context_layer1)
        context_layer0 = self.upsample0_bn(context_layer0)
        context_layer0 = self.relu(context_layer0)
        
        bs = images.size(0)
        
        
        inference_out = self.diffusion.ddim_sample_new(self.diffusion_model, inference_steps, start_steps, shape=(bs,) + self.hm_size, context=[context_layer0, context_layer1, context_layer2, context_layer3], 
                                                                         eta=self.ddim_eta, x_T=inter_sample, annt_gen=annt_gen)
        
        return inference_out, hx


       