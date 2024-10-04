import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from model.model_base import Bottleneck, BottleneckConvLSTM

class ModelSpatial(nn.Module):
    # Define a ResNet 50-ish arch
    def __init__(self, args):
        # Resnet Feature Extractor

        block = Bottleneck
        layers_scene = [3, 4, 6, 3, 2]
        layers_face = [3, 4, 6, 3, 2]
        self.inplanes_scene = 64
        self.inplanes_face = 64
        super(ModelSpatial, self).__init__()
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

        # encoding for saliency
        self.compress_conv1 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1 = nn.BatchNorm2d(1024)
        self.compress_conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2 = nn.BatchNorm2d(512)

        # encoding for in/out
        self.compress_conv1_inout = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1_inout = nn.BatchNorm2d(512)
        self.compress_conv2_inout = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2_inout = nn.BatchNorm2d(1)
        self.fc_inout = nn.Linear(49, 1)

        # decoding
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.deconv_bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.deconv_bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2)
        self.deconv_bn3 = nn.BatchNorm2d(1)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=1, stride=1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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

        # scene + face feat -> in/out
        encoding_inout = self.compress_conv1_inout(scene_face_feat)
        encoding_inout = self.compress_bn1_inout(encoding_inout)
        encoding_inout = self.relu(encoding_inout)
        encoding_inout = self.compress_conv2_inout(encoding_inout)
        encoding_inout = self.compress_bn2_inout(encoding_inout)
        encoding_inout = self.relu(encoding_inout)
        encoding_inout = encoding_inout.view(-1, 49)
        encoding_inout = self.fc_inout(encoding_inout)

        # scene + face feat -> encoding -> decoding
        encoding = self.compress_conv1(scene_face_feat)
        encoding = self.compress_bn1(encoding)
        encoding = self.relu(encoding)
        encoding = self.compress_conv2(encoding)
        encoding = self.compress_bn2(encoding)
        encoding = self.relu(encoding)

        x = self.deconv1(encoding)
        x = self.deconv_bn1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.deconv_bn2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.deconv_bn3(x)
        x = self.relu(x)
        x = self.conv4(x)

        return x, encoding_inout


class ModelSpatial_GradCAM(nn.Module):
    # Define a ResNet 50-ish arch
    def __init__(self, args):
        # Resnet Feature Extractor

        block = Bottleneck
        layers_scene = [3, 4, 6, 3, 2]
        layers_face = [3, 4, 6, 3, 2]
        self.inplanes_scene = 64
        self.inplanes_face = 64
        super(ModelSpatial_GradCAM, self).__init__()
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
        self.attn = nn.Linear(2064, 1*7*7)

        # encoding for saliency
        self.compress_conv1 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1 = nn.BatchNorm2d(1024)
        self.compress_conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2 = nn.BatchNorm2d(512)

        # encoding for in/out
        self.compress_conv1_inout = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1_inout = nn.BatchNorm2d(512)
        self.compress_conv2_inout = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2_inout = nn.BatchNorm2d(1)
        self.fc_inout = nn.Linear(49, 1)

        # decoding
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.deconv_bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.deconv_bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2)
        self.deconv_bn3 = nn.BatchNorm2d(1)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=1, stride=1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        images, head, face, gradcam = input
        
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
        
        gradcam_reduced = self.maxpool(self.maxpool(gradcam)).view(-1, 256)
        attn_weights = self.attn(torch.cat((head_reduced, gradcam_reduced, face_feat_reduced), 1))

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

        # scene + face feat -> in/out
        encoding_inout = self.compress_conv1_inout(scene_face_feat)
        encoding_inout = self.compress_bn1_inout(encoding_inout)
        encoding_inout = self.relu(encoding_inout)
        encoding_inout = self.compress_conv2_inout(encoding_inout)
        encoding_inout = self.compress_bn2_inout(encoding_inout)
        encoding_inout = self.relu(encoding_inout)
        encoding_inout = encoding_inout.view(-1, 49)
        encoding_inout = self.fc_inout(encoding_inout)

        # scene + face feat -> encoding -> decoding
        encoding = self.compress_conv1(scene_face_feat)
        encoding = self.compress_bn1(encoding)
        encoding = self.relu(encoding)
        encoding = self.compress_conv2(encoding)
        encoding = self.compress_bn2(encoding)
        encoding = self.relu(encoding)

        x = self.deconv1(encoding)
        x = self.deconv_bn1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.deconv_bn2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.deconv_bn3(x)
        x = self.relu(x)
        x = self.conv4(x)


        return x, encoding_inout