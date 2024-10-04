import torch
import numpy as np
import os, sys
import pickle
import cv2
import random
import copy
import pdb
import traceback
import h5py
import spacy
import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.caption_utils import construct_sample_caption, construct_sample_vqa
sys.path.insert(0, (os.path.join("/home/qiaomu/code/gaze/mulmod", "OFA")) )
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from fairseq import utils, tasks
from fairseq import checkpoint_utils
from fairseq import distributed_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from models.vqa_custom_model import Custom_Model, GradCAM_model
from utils.eval_utils import eval_step
from utils.zero_shot_utils import zero_shot_step
from utils.vqa_utils import Utils_Obj
from tasks.mm_tasks.vqa_gen import VqaGenTask
from models.ofa import OFAModel
from PIL import Image
from torchvision import transforms
from utils.eval_utils import eval_caption
from utils.gradcam_utils import *

def plot_gradcam_alllayers(image, grad_cams, save_path, plot_img_size=(224,224)):
    fig = plt.figure(figsize=(12,4))
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
    

data_dir = '/nfs/bigrod/add_disk0/qiaomu/datasets/gaze/videoattentiontarget'
image_dir = os.path.join(data_dir, 'images')
shows = glob.glob(os.path.join(data_dir, 'annotations_body', 'train', '*'))
all_sequence_paths = []
for s in shows:
    sequence_annotations = glob.glob(os.path.join(s, '*', '*.txt'))  # each annotation file is for one person
    all_sequence_paths.extend(sequence_annotations)
save_dir = os.path.join(data_dir, 'gradcams_body_new', 'train')

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
#with open('/data/add_disk0/qiaomu/datasets/gaze/gazefollow/all_word')
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

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
plot_dpi = 80
nlp = spacy.load("en_core_web_sm")
try:
    for idx, sequence_path in enumerate(tqdm(all_sequence_paths)):
        df = pd.read_csv(sequence_path, header=None, index_col=False,
                         names=['path', 'xmin', 'ymin', 'xmax', 'ymax', 'gazex', 'gazey', 'body_x1', 'body_y1', 'body_x2', 'body_y2'])
        person_name = sequence_path.split('/')[-1].split('.')[0]
        show_name = sequence_path.split('/')[-3]
        clip = sequence_path.split('/')[-2]
        seq_len = len(df.index)
        this_plot_dir = os.path.join(save_dir, "plots", show_name, clip)
        this_gradcam_dir = os.path.join(save_dir, "gradcams", show_name, clip)
        if not os.path.exists(this_plot_dir):
            os.makedirs(this_plot_dir)
        if not os.path.exists(this_gradcam_dir):
            os.makedirs(this_gradcam_dir)
        
        data_save = {}        
        for i, row in df.iterrows():
            img_name, head_x1, head_y1, head_x2, head_y2, gaze_x, gaze_y = row[['path', 'xmin', 'ymin', 'xmax', 'ymax', 'gazex', 'gazey']]
            body_x1, body_y1, body_x2, body_y2 = row[['body_x1', 'body_y1', 'body_x2', 'body_y2']]
            head_box = [head_x1, head_y1, head_x2, head_y2]
            if gaze_x == -1 and gaze_y == -1:
                inout = False
            else:
                inout =True

            img_path = os.path.join(image_dir, show_name, clip, img_name)
            imgname = img_name.split('.')[0]
            
            #if inout:    
            new_imgname = imgname + '_'+ person_name
            save_path = os.path.join(this_plot_dir, new_imgname+'.jpg')
            # get gradcam for image
            image = Image.open(img_path)
            #plt_person(image, body_box, save_path)
            img = cv2.imread(img_path)
            start_pt, end_pt = (body_x1, body_y1), (body_x2, body_y2)
            #start_pt, end_pt = (int(head_box[0]), int(head_box[1])), (int(head_box[2]), int(head_box[3]))
            cv2.rectangle(img, start_pt, end_pt, (0,255,0), thickness=2)
            cv2.imwrite(save_path, img)
            
            image_new = Image.open(save_path)
            
            question='What is the person in the green bounding box looking at?'
            sample, result = utils_obj.get_vqa_result(image_new, question, full_model)
            
            
            new_result = utils_obj.get_str_from_tokens(result)
            first_res = new_result[0]
            first_word = first_res['str'].split()[0]
            
            #if first_word.lower()=='the' or first_word.lower()=='a' or first_word.lower()=='an':
                #gradcam_step = len(first_res['tokens']) - 2
            #else:
                #gradcam_step = 0
            doc = nlp(first_res['str'].strip()) 
            all_steps = []
            for token_idx, token in enumerate(doc):
                if token.pos_.lower()=='noun' or token.pos_.lower()=='propn' or token.pos_.lower()=='pron':
                    all_steps.append(token_idx)
                else:
                    if len(all_steps)>0:
                        # allow continuous nouns, but if another type jumps in, then stop
                        break
            
            if len(all_steps)==0:
                all_steps.append(len(first_res['tokens']) - 2) 
            else:
                all_steps = [all_steps[-1]]
                #all_steps = [all_steps[0]]
            #gradcam_step = len(first_res['tokens']) - 2
            print((show_name, new_imgname))
            print(first_res['str'])
            print(first_res['tokens'])
            print(all_steps)
            # get gradcam and visualize
            for param in gradcam_model.model.encoder.embed_images.parameters():
                param.requires_grad=False
            gradcams_allsteps, gradcams_resize_all = [],[] 
            
            for gradcam_step in all_steps:
                if gradcam_step > len(first_res['tokens']):
                    gradcam_step = len(first_res['tokens']) - 2
                gradcams, gradcams_resize  = get_gradcam_on_attweights(result[0], sample, gradcam_model, tgt_layer_idx='all', step=gradcam_step, bos=full_model.bos, eos=full_model.eos)
                gradcams_allsteps.append(gradcams)
                gradcams_resize_all.append(gradcams_resize)
                word = doc[min(gradcam_step, len(doc)-1)]
                    
                #save_path = os.path.join(save_dir, f'{imgname}_gradcam.png')
                #plot_gradcam_alllayers(image_new, gradcams_resize, save_path, plot_img_size=(224,224))
            
            save_path = os.path.join(this_plot_dir, f'{new_imgname}_gradcam.png')
            plot_gradcam_alllayers(image_new, gradcams_resize, save_path, plot_img_size=(224,224))
                
                        
            #save_path = os.path.join(save_dir, f'{imgname}_head.png')
            #plt_person(image, head_box, save_path, color=(1,0,0))
            if imgname not in data_save:
                data_save[imgname] = {}
            

            data_save[imgname]['answer'] = first_res['str']
            data_save[imgname]['tokens'] = first_res['tokens']
            data_save[imgname]['9th_layer'] = gradcams[-3]
            data_save[imgname]['10th_layer'] = gradcams[-2]
            data_save[imgname]['11th_layer'] = gradcams[-1]
        
        with open(os.path.join(this_gradcam_dir, person_name+'.pkl'), 'wb') as file:
            print(f"Saving: {show_name}")
            pickle.dump(data_save, file)

    #f = h5py.File(os.path.join(gf_data_dir, "person_detections",'gradcams_train_ofa_2.h5'), 'w')
    #s = str(data_save)
    #for grp_name in data_save:
        #grp = f.create_group(grp_name)
        #for dset_name in data_save[grp_name]:
            #dset_name_str = str(dset_name)
                #dset = grp.create_dataset(dset_name_str, data = data_save[grp_name][dset_name])
    #f.close()
            
except Exception: 
    print(traceback.format_exc())
    pdb.set_trace()