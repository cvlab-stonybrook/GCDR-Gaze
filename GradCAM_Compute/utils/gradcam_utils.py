import torch
import copy
import numpy as np
import torch.nn.functional as F

def get_gradcam_layer(gradcam_model, layer_idx, tgt_size=224, imgout_size=30):
    num_img_tokens = imgout_size * imgout_size
    decoder_att = gradcam_model.model.decoder.layers[layer_idx].encoder_attn.attn_probs
    grad = gradcam_model.model.decoder.layers[layer_idx].encoder_attn.attn_gradients
    num_heads = grad.size(0)
    grad, decoder_att = grad[:,-1,:num_img_tokens], decoder_att[:,-1,:num_img_tokens]
    grad, decoder_att = grad.reshape((num_heads, imgout_size,imgout_size)), decoder_att.reshape((num_heads, imgout_size,imgout_size))
    gradcam = grad * decoder_att
    gradcam = torch.clamp(gradcam, min=0).mean(dim=0).unsqueeze(0).unsqueeze(0)
    gradcam_resize = torch.nn.functional.interpolate(gradcam, size=tgt_size, mode='bilinear').squeeze()
    gradcam_resize_norm = (gradcam_resize - gradcam_resize.min()) / (gradcam_resize.max() - gradcam_resize.min())
    if torch.isnan(gradcam_resize_norm).any():
        import pdb
        pdb.set_trace()
    gradcam, gradcam_resize_norm = gradcam.squeeze().detach().cpu().numpy(), gradcam_resize_norm.detach().cpu().numpy()
    return gradcam, gradcam_resize_norm

def get_gradcampp_layer(gradcam_model, layer_idx, tgt_size=224, imgout_size=30, scores=None):
    num_img_tokens = imgout_size * imgout_size
    decoder_att = gradcam_model.model.decoder.layers[layer_idx].encoder_attn.attn_probs
    grad = gradcam_model.model.decoder.layers[layer_idx].encoder_attn.attn_gradients
    num_heads = grad.size(0)
    grad, decoder_att = grad[:,-1,:num_img_tokens], decoder_att[:,-1,:num_img_tokens]
    grad, decoder_att = grad.reshape((num_heads, 1, imgout_size,imgout_size)), decoder_att.reshape((num_heads, 1, imgout_size,imgout_size))
    
    b, k, u, v = grad.size()
    alpha_num = grad.pow(2)
    global_sum = decoder_att.view(b, k, u*v).sum(-1, keepdim=True).view(b, k, 1, 1) 
    alpha_denom = grad.pow(2).mul(2) + global_sum.mul(grad.pow(3))
    alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
    alpha = alpha_num.div(alpha_denom+1e-7)
    positive_gradients = F.relu(grad) # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
    weights = (alpha*positive_gradients).view(b, k, u*v).sum(-1).view(b, k, 1, 1)
    
    gradcam = (weights*decoder_att).sum(1)
    gradcam = torch.clamp(gradcam, min=0).mean(dim=0).unsqueeze(0).unsqueeze(0)
    gradcam_resize = torch.nn.functional.interpolate(gradcam, size=tgt_size, mode='bilinear').squeeze()
    gradcam_resize_norm = (gradcam_resize - gradcam_resize.min()) / (gradcam_resize.max() - gradcam_resize.min())
    if torch.isnan(gradcam_resize_norm).any():
        import pdb
        pdb.set_trace()
    gradcam, gradcam_resize_norm = gradcam.squeeze().detach().cpu().numpy(), gradcam_resize_norm.detach().cpu().numpy()
    return gradcam, gradcam_resize_norm


def get_gradcam_on_attweights(result, ori_sample, gradcam_model, tgt_layer_idx=11, step=0, bos=0, eos=2, imgout_size=30):
    result_tokens = result["tokens"].clone()
    tgt_index = result_tokens[step].item()
    input_tokens = torch.tensor([bos]).to(result_tokens)
    input_tokens = torch.cat([input_tokens, result_tokens[:step]]).unsqueeze(0)
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
        gradcam, gradcam_resize = get_gradcam_layer(gradcam_model, tgt_layer_idx, imgout_size=imgout_size)
        gradcams.append(gradcam)
        gradcams_resize.append(gradcam_resize)
    elif tgt_layer_idx=='all':
        num_layers = gradcam_model.model.decoder.num_layers
        for layer_idx in range(num_layers-3, num_layers):
            gradcam, gradcam_resize = get_gradcam_layer(gradcam_model, layer_idx, imgout_size=imgout_size)
            gradcams.append(gradcam)
            gradcams_resize.append(gradcam_resize)
    return gradcams, gradcams_resize



def get_gradcampp_on_attweights(result, ori_sample, gradcam_model, tgt_layer_idx=11, step=0, bos=0, eos=2, imgout_size=30):
    result_tokens = result["tokens"].clone()
    tgt_index = result_tokens[step].item()
    input_tokens = torch.tensor([bos]).to(result_tokens)
    input_tokens = torch.cat([input_tokens, result_tokens[:step]]).unsqueeze(0)
    sample = copy.deepcopy(ori_sample)
    sample['tgt_tokens'] = input_tokens
    gradcam_model.zero_grad()
    return_att_layer = tgt_layer_idx if type(tgt_layer_idx)==int else 11
    logits_out, encoder_out_tensor, decoder_att = gradcam_model(sample, decoder_alignment_layer=return_att_layer, avg_attn_heads=False, get_decoder_attgrad=True)
    loss = logits_out[:, tgt_index].sum()
    loss.backward()
    gradcams, gradcams_resize = [],[]
    if type(tgt_layer_idx)==int:
        gradcam, gradcam_resize = get_gradcampp_layer(gradcam_model, tgt_layer_idx, imgout_size=imgout_size, scores=logits_out)
        gradcams.append(gradcam)
        gradcams_resize.append(gradcam_resize)
    elif tgt_layer_idx=='all':
        num_layers = gradcam_model.model.decoder.num_layers
        for layer_idx in range(num_layers-3, num_layers):
            gradcam, gradcam_resize = get_gradcampp_layer(gradcam_model, layer_idx, imgout_size=imgout_size, scores=logits_out)
            gradcams.append(gradcam)
            gradcams_resize.append(gradcam_resize)
    return gradcams, gradcams_resize