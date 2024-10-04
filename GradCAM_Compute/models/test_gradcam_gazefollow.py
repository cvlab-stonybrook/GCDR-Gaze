import torch
import numpy as np
import re
import os
import cv2
import copy
from fairseq import utils,tasks
from fairseq import checkpoint_utils
from fairseq import distributed_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from utils.zero_shot_utils import zero_shot_step
from utils.vqa_utils import Utils_Obj
from tasks.mm_tasks.vqa_gen import VqaGenTask
from models.vqa_custom_model import Custom_Model, GradCAM_model
from PIL import Image
from IPython.core.debugger import Pdb
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import ImageGrid
from skimage.transform import resize

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms



def get_gradcam_layer(gradcam_model, layer_idx, tgt_size=224, num_img_tokens=900):
    decoder_att = gradcam_model.model.decoder.layers[layer_idx].encoder_attn.attn_probs
    grad = gradcam_model.model.decoder.layers[layer_idx].encoder_attn.attn_gradients
    num_heads = grad.size(0)
    grad, decoder_att = grad[:,-1,:num_img_tokens], decoder_att[:,-1,:num_img_tokens]
    grad, decoder_att = grad.reshape((num_heads, 30,30)), decoder_att.reshape((num_heads, 30,30))
    gradcam = grad * decoder_att
    gradcam = torch.clamp(gradcam, min=0).mean(dim=0).unsqueeze(0).unsqueeze(0)
    gradcam_resize = torch.nn.functional.interpolate(gradcam, size=tgt_size, mode='bilinear').squeeze()
    gradcam_resize = (gradcam_resize - gradcam_resize.min()) / (gradcam_resize.max() - gradcam_resize.min())
    gradcam, gradcam_resize = gradcam.squeeze().detach().cpu().numpy(), gradcam_resize.detach().cpu().numpy()
    return gradcam, gradcam_resize


def get_gradcam_on_attweights(result, ori_sample, gradcam_model, tgt_layer_idx=11, step=0, bos=0, eos=2):
    result_tokens = result["tokens"].clone()
    tgt_index = result_tokens[step].item()
    input_tokens = torch.tensor([bos]).to(result_tokens)
    input_tokens = torch.cat([input_tokens, result_tokens[:step]]).unsqueeze(0)
    print(input_tokens)
    sample = copy.deepcopy(ori_sample)
    sample['tgt_tokens'] = input_tokens
    gradcam_model.zero_grad()
    return_att_layer = tgt_layer_idx if type(tgt_layer_idx)==int else 11
    logits_out, encoder_out_tensor, decoder_att = gradcam_model(sample, decoder_alignment_layer=return_att_layer, avg_attn_heads=False, get_decoder_attgrad=True)
    num_slots = logits_out.size(1)
    one_hot = np.zeros((1, num_slots))
    one_hot[0, tgt_index] = 1
    one_hot = torch.tensor(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_out)
    one_hot.backward()
    gradcams, gradcams_resize = [],[]
    if type(tgt_layer_idx)==int:
        gradcam, gradcam_resize = get_gradcam_layer(gradcam_model, tgt_layer_idx)
        gradcams.append(gradcam)
        gradcams_resize.append(gradcam_resize)
    elif tgt_layer_idx=='all':
        for layer_idx in range(gradcam_model.model.decoder.num_layers):
            gradcam, gradcam_resize = get_gradcam_layer(gradcam_model, layer_idx)
            gradcams.append(gradcam)
            gradcams_resize.append(gradcam_resize)
    return gradcams, gradcams_resize


def plot_gradcam_alllayers(image, grad_cams, plot_img_size=(224,224)):
    fig = plt.figure(figsize=(18, 4.5))
    width, height = image.size
    image.convert('RGB')
    image = image.resize(plot_img_size, resample=Image.BILINEAR)
    image = np.asarray(image)
    for i in range(9,12):
        grad_cam = grad_cams[i]
        ax = fig.add_subplot(1,3,i-8)
        ax = plt.gca()
        ax.axis('off')
        ax.imshow(image)
        ax.imshow(grad_cam, cmap='jet', vmin=0.0, vmax=1.0, alpha=0.3)
        ax.set_title(f"layer_{i}")
    plt.savefig('./figures/compare.png')
    plt.show()
    plt.close()


if __name__=='__main__':
    # Register VQA task
    tasks.register_task('vqa_gen',VqaGenTask)
    # turn on cuda if GPU is available
    use_cuda = torch.cuda.is_available()
    # use fp16 only when GPU is available
    use_fp16 = False

    # specify some options for evaluation
    parser = options.get_generation_parser()
    input_args = ["", "--task=vqa_gen", "--beam=100", "--unnormalized", "--path=/nfs/bigquery/add_disk0/qiaomu/ckpts/multimodal/ofa/ofa_large.pt", "--bpe-dir=utils/BPE"]
    args = options.parse_args_and_arch(parser, input_args)
    cfg = convert_namespace_to_omegaconf(args)
    os.environ['CUDA_VISIBLE_DEVICES']='3'
    # Load pretrained ckpt & config
    task = tasks.setup_task(cfg.task)
    models, cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        task=task
    )
    # Move models to GPU
    for model in models:
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Initialize generator
    generator = task.build_generator(models, cfg.generation) 
    model = models[0]
    tgt_dict = task.target_dictionary
    full_model = Custom_Model(cfg.generation, model, tgt_dict)
    gradcam_model = GradCAM_model(copy.deepcopy(model))

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((cfg.task.patch_image_size, cfg.task.patch_image_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Text preprocess
    bos_item = torch.LongTensor([task.src_dict.bos()])
    eos_item = torch.LongTensor([task.src_dict.eos()])
    pad_idx = task.src_dict.pad()
    utils_obj = Utils_Obj(task, patch_resize_transform)
    
    
    image_path = ''
    # get gradcam for image
    image = Image.open(image_path)
    question=''
    sample, result = utils_obj.get_vqa_result(image, question, full_model) 
    new_result = utils_obj.get_str_from_tokens(result, task, generator)
 
    for param in gradcam_model.model.encoder.embed_images.parameters():
        param.requires_grad=False
    
    gradcams, gradcams_resize  = get_gradcam_on_attweights(result[0], sample, gradcam_model, tgt_layer_idx='all', step=0, bos=full_model.bos, eos=full_model.eos)
    print(new_result)
    plot_gradcam_alllayers(image, gradcams_resize,  plot_img_size=(224,224))