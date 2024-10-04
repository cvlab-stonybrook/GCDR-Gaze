import torch
import numpy as np
import os, sys
import pickle
import cv2
import random
import copy
import pdb
import traceback
import argparse
import spacy
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.gazefollow_dataset import GazeFollow_body_head
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from fairseq import utils, tasks
from fairseq import checkpoint_utils
from fairseq import distributed_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from models.vqa_custom_model import Custom_Model, GradCAM_model
from utils.vqa_utils import Utils_Obj
from tasks.mm_tasks.vqa_gen import VqaGenTask
from PIL import Image
from torchvision import transforms
from utils.gradcam_utils import *

def plot_gradcam_alllayers(image, grad_cams, save_path, plot_img_size=(224,224)):
    fig = plt.figure(figsize=(14,4))
    width, height = image.size
    image.convert('RGB')
    image = image.resize(plot_img_size, resample=Image.BILINEAR)
    image = np.asarray(image)
    for i in range(3):
        grad_cam = grad_cams[i]
        ax = fig.add_subplot(1,3,i+1)
        ax = plt.gca()
        ax.axis('off')
        ax.imshow(image)
        ax.imshow(grad_cam, cmap='jet', vmin=0.0, vmax=1.0, alpha=0.3)
        ax.set_title(f"layer_{9+i}")
    plt.savefig(save_path)
    plt.close()

def plt_person(img, new_box, save_path, plot_dpi = 80, color=(0,1,0)):
    width, height = img.size
    img = np.array(img)
    fig = plt.figure(figsize=(width/plot_dpi, height/plot_dpi), dpi=plot_dpi)
    plt.axis('off')
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    rect = patches.Rectangle((int(new_box[0]), int(new_box[1])), int(new_box[2]-new_box[0]), int(new_box[3]-new_box[1]), linewidth=3, edgecolor=color, facecolor='none')
    plt.imshow(img)
    ax = plt.gca()
    ax.add_patch(rect)
    #plt.show()
    plt.savefig(save_path, pad_inches=0, bbox_inches='tight')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate gradcam for GazeFollow')
    parser.add_argument('--base_dir', type=str, default='/data/add_disk0/qiaomu/datasets/gaze/gazefollow')
    parser.add_argument('--vqa_weights', type=str, default='/nfs/bigquery/add_disk0/qiaomu/ckpts/multimodal/ofa/ofa_large.pt')
    parser.add_argument('--vis_gradcam', action='store_true')
    parser.add_argument('--save_folder', type=str, default='gradcam_gazefollow')
    parser.add_argument('--sample_num', type=int, default=-1)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    gf_data_dir = args.base_dir

    tasks.register_task('vqa_gen',VqaGenTask)
    # turn on cuda if GPU is available
    use_cuda = torch.cuda.is_available()
    # use fp16 only when GPU is available
    use_fp16 = False

    # specify some options for evaluation
    parser = options.get_generation_parser()
    input_args = ["", "--task=vqa_gen", "--beam=100", "--unnormalized", f"--path={args.vqa_weights}", "--bpe-dir=utils/BPE"]
    opt = options.parse_args_and_arch(parser, input_args)
    cfg = convert_namespace_to_omegaconf(opt)
    # # Load Model
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

    # Initialize generator
    generator = task.build_generator(models, cfg.generation)
    # Text preprocess
    bos_item = torch.LongTensor([task.src_dict.bos()])
    eos_item = torch.LongTensor([task.src_dict.eos()])
    pad_idx = task.src_dict.pad()
    utils_obj = Utils_Obj(task, generator, patch_resize_transform)

    dataset = GazeFollow_body_head(gf_data_dir, test=args.test)
    if args.sample_num>0:
        sample_idx = random.sample(list(range(len(dataset))), args.sample_num)
        dataset = torch.utils.data.Subset(dataset, sample_idx)    

    dataloader = DataLoader(dataset, batch_size=1, num_workers = 0, shuffle=False)

    save_dir = os.path.join(gf_data_dir, "visualizations", args.save_folder)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    plot_dpi = 80
    nlp = spacy.load("en_core_web_sm")
    data_save = {}
    try:
        for idx, data in enumerate(tqdm(dataloader)):
            img_path, eye_x, body_box, head_box, inout, detected = data
            img_path, body_box, head_box = img_path[0], body_box[0].numpy(), head_box[0].numpy()
            eye_x = eye_x[0].item()
            imgpath = os.path.split(img_path)[-1]
            imgname = imgpath.split('.')[0]
            save_path = os.path.join(save_dir, imgpath)
            # get gradcam for image
            image = Image.open(img_path)
            img = cv2.imread(img_path)
            start_pt, end_pt = (body_box[0], body_box[1]), (body_box[2], body_box[3])
            #start_pt, end_pt = (int(head_box[0]), int(head_box[1])), (int(head_box[2]), int(head_box[3]))
            cv2.rectangle(img, start_pt, end_pt, (0,255,0), thickness=2)
            #cv2.imwrite(save_path, img)
            
            # generate gradcam for all, irrespective or in or out
            #image_new = Image.open(save_path)
            image_new = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if detected[0]:
                question='What is the person in the green bounding box looking at?'
                sample, result = utils_obj.get_vqa_result(image_new, question, full_model)
            else:
                question='What is the person looking at?'
                sample, result = utils_obj.get_vqa_result(image, question, full_model)
                
            new_result = utils_obj.get_str_from_tokens(result)
            first_res = new_result[0]
            first_word = first_res['str'].split()[0]
            
            doc = nlp(first_res['str'].strip()) 
            all_steps = []
            for token_idx, token in enumerate(doc):
                if token.pos_.lower()=='noun' or token.pos_.lower()=='propn' or token.pos_.lower()=='pron':
                    all_steps.append(token_idx)
                else:
                    if len(all_steps)>0:
                        # select the first noun (or noun phrase): allow continuous nouns, but if another type jumps in, then stop
                        break
            
            if len(all_steps)==0:
                gradcam_step = len(first_res['tokens']) - 2
            else:
                gradcam_step = all_steps[-1]  # select the last noun in continuous nouns: e.g. tennis racket
            print(img_path)
            print(first_res['str'])
            print(first_res['tokens'])
            print(gradcam_step)
            # get gradcam and visualize
            for param in gradcam_model.model.encoder.embed_images.parameters():
                param.requires_grad=False
            gradcams_allsteps, gradcams_resize_all = [],[] 
            
            if gradcam_step > len(first_res['tokens']):
                gradcam_step = len(first_res['tokens']) - 2
            # original size: 30
            gradcams, gradcams_resize  = get_gradcam_on_attweights(result[0], sample, gradcam_model, tgt_layer_idx='all', step=gradcam_step, bos=full_model.bos, eos=full_model.eos, imgout_size=30)
            gradcams_allsteps.append(gradcams)
            gradcams_resize_all.append(gradcams_resize)
            word = doc[min(gradcam_step, len(doc)-1)]
                
            save_path = os.path.join(save_dir, f'{imgname}_gradcam.png')
            if args.vis_gradcam:
                plot_gradcam_alllayers(image_new, gradcams_resize, save_path, plot_img_size=(224,224))
                
            if imgpath not in data_save:
                data_save[imgpath] = {}
            data_save[imgpath][eye_x] = {}
            data_save[imgpath][eye_x]['answer'] = first_res['str']
            data_save[imgpath][eye_x]['tokens'] = first_res['tokens']

            data_save[imgpath][eye_x]['9th_layer'] = gradcams[-3]
            data_save[imgpath][eye_x]['10th_layer'] = gradcams[-2]
            data_save[imgpath][eye_x]['11th_layer'] = gradcams[-1]
        
        if args.sample_num == -1:
            # entire dataset
            with open(os.path.join(save_dir, "gradcams_train_person.pkl"), 'wb') as file:
                pickle.dump(data_save, file)
            
    except Exception: 
        print(traceback.format_exc())
        pdb.set_trace()