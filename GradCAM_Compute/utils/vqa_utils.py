import torch
import re
import random
import string
import numpy as np
from PIL import Image
from fairseq import utils
# Normalize the question




class Utils_Obj:
    def __init__(self, task, generator, image_transform):
        self.task = task
        self.generator = generator
        self.image_transform = image_transform
        self.bos_item = torch.LongTensor([task.src_dict.bos()])
        self.eos_item = torch.LongTensor([task.src_dict.eos()])
        self.pad_idx = task.src_dict.pad()

    def pre_question(self, question, max_ques_words):
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

    def encode_text(self, text, length=None, append_bos=False, append_eos=False):
        s = self.task.tgt_dict.encode_line(
            line=self.task.bpe.encode(text),
            add_if_not_exist=False,
            append_eos=False
        ).long()
        if length is not None:
            s = s[:length]
        if append_bos:
            s = torch.cat([self.bos_item, s])
        if append_eos:
            s = torch.cat([s, self.eos_item])
        return s

    # Construct input for open-domain VQA task
    def construct_sample(self, image: Image, question: str):
        patch_image = self.image_transform(image).unsqueeze(0)
        patch_mask = torch.tensor([True])
        question = self.pre_question(question, self.task.cfg.max_src_length)
        question = question + '?' if not question.endswith('?') else question
        #Pdb().set_trace()
        src_text = self.encode_text(' {}'.format(question), append_bos=True, append_eos=True).unsqueeze(0)

        src_length = torch.LongTensor([s.ne(self.pad_idx).long().sum() for s in src_text])
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

    def construct_sample_caption(self, image: Image):
        patch_image = self.image_transform(image).unsqueeze(0)
        patch_mask = torch.tensor([True])
        src_text = self.encode_text(" what does the image describe?", append_bos=True, append_eos=True).unsqueeze(0)
        src_length = torch.LongTensor([s.ne(self.pad_idx).long().sum() for s in src_text])
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
    
    def decode_fn(self, x, tokenizer=None):
        tgt_dict, bpe = self.task.tgt_dict, self.task.bpe
        x = tgt_dict.string(x.int().cpu(), extra_symbols_to_ignore={self.generator.bos, self.generator.eos})
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x
    
    def eval_caption(self, models, sample, **kwargs):
        transtab = str.maketrans({key: None for key in string.punctuation})
        hypos = self.task.inference_step(self.generator, models, sample)
        results = []
        for i, sample_id in enumerate(sample["id"].tolist()):
            detok_hypo_str = self.decode_fn(hypos[i][0]["tokens"])
            results.append({"image_id": str(sample_id), "caption": detok_hypo_str.translate(transtab).strip()})
        return results, None

    def get_str_from_tokens(self, result):
        info = []
        for idx, beam_out in enumerate(result):
            tokens = beam_out["tokens"]
            detok_hypo_str = self.decode_fn(tokens)
            info.append({"tokens":tokens.cpu(), "str": detok_hypo_str, "score":beam_out["score"]})
        return info 

    def get_vqa_result(self, image, question, model, use_cuda=True, use_fp16=False):
        sample = self.construct_sample(image, question)
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        sample = utils.apply_to_sample(self.apply_half, sample) if use_fp16 else sample
        # Run eval step for open-domain VQA
        with torch.no_grad():
            result = model(sample["net_input"])[0]
        return sample, result
    
    # Function to turn FP32 to FP16
    def apply_half(self, t):
        if t.dtype is torch.float32:
            return t.to(dtype=torch.half)
        return t
    
    
    def get_caption(self, models, path, box, use_fp16=False, use_cuda=True):
        img = Image.open(path)
        img_w, img_h = img.size
        x1,y1,x2,y2 = box
        width, height = (x2-x1)/2, (y2-y1)/2  # half width, half height
        x_cen, y_cen = int((x1+x2)/2), int((y1+y2)/2)
        x1_exp, x2_exp = round(np.clip(x_cen-width*1.1, 0, img_w-1)), round(np.clip(x_cen+width*1.1, 0, img_w-1))
        y1_exp, y2_exp = round(np.clip(y_cen-height*1.1, 0, img_h-1)), round(np.clip(y_cen+height*1.1, 0, img_h-1))
        new_box = np.array([x1_exp, y1_exp, x2_exp, y2_exp])
        person_crop = img.copy().crop((x1_exp, y1_exp, x2_exp, y2_exp))
        sample = self.construct_sample_caption(person_crop)
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        sample = utils.apply_to_sample(self.apply_half, sample) if use_fp16 else sample
        with torch.no_grad():
            result, scores = self.eval_caption( models, sample)
        
        return img, result[0]['caption'], new_box

    
    


    
