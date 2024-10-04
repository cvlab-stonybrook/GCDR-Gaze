import torch
from PIL import Image
import os,sys
import re
import numpy as np
sys.path.insert(0, (os.path.join(os.path.dirname(os.path.abspath(__file__)), "OFA")) )
from fairseq import utils,tasks

# Normalize the question
def pre_question(question, max_ques_words):
    question = question.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ')
    question = re.sub(
        r"\s{2,}",
        ' ',
        question,
    )
    question = question.rstrip('\n')
    question = question.strip(' ')
    # truncate question
    question_words = question.split(' ')
    if len(question_words) > max_ques_words:
        question = ' '.join(question_words[:max_ques_words])
    return question

def encode_text(task, text, length=None, append_bos=False, append_eos=False):
    s = task.tgt_dict.encode_line(
        line=task.bpe.encode(text),
        add_if_not_exist=False,
        append_eos=False
    ).long()
    bos_item = torch.LongTensor([task.src_dict.bos()])
    eos_item = torch.LongTensor([task.src_dict.eos()])
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    return s

# Construct input for open-domain VQA task
def construct_sample_vqa(task, image_transform, image: Image, question: str):
    patch_image = image_transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])
    pad_idx = task.src_dict.pad()

    question = pre_question(question, task.cfg.max_src_length)
    question = question + '?' if not question.endswith('?') else question
    src_text = encode_text(task, ' {}'.format(question), append_bos=True, append_eos=True).unsqueeze(0)

    src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
    ref_dict = np.array([{'yes': 1.0}]) # just placeholder
    sample = {
        "id":np.array(['42']),
        "net_input": {
            "src_tokens": src_text,
            "src_lengths": src_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask,
        },
        "ref_dict": ref_dict,
    }
    return sample

def construct_sample_caption(task, image_transform, image: Image):
    patch_image = image_transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])
    pad_idx = task.src_dict.pad()
    src_text = encode_text(task, " what does the image describe?", append_bos=True, append_eos=True).unsqueeze(0)
    src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
    sample = {
        "id":np.array(['42']),
        "net_input": {
            "src_tokens": src_text,
            "src_lengths": src_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask
        }
    }
    return sample


# Function to turn FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t
