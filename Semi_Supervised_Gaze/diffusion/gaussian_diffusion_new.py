"""
copied from https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/gaussian_diffusion.py
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math
import pdb
import numpy as np
import torch
from torch import nn
from .ddim_utils import make_ddim_timesteps, make_ddim_sampling_parameters, noise_like
from time import time

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        #scale = 1.0  # modified by miao
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL

def to_torch(array, device='cpu'):
    return torch.tensor(array).float().to(device)

def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        schedule='linear',
        rescale_timesteps=False,
        steps = 1000,
        denoise_steps = -1,
        scale=1.0,
        pred_noise = False,
        sigma_small=True,
        scale_input = False,
        ddim_eta = 0.0,
        annt_weight_temp=1.0,
    ):  
        super(GaussianDiffusion, self).__init__()
        
        if sigma_small:
            self.model_var_type = ModelVarType.FIXED_SMALL
        else:
            self.model_var_type = ModelVarType.FIXED_LARGE
        self.rescale_timesteps = rescale_timesteps
        self.pred_noise = pred_noise
        self.scale = scale
        self.scale_input = scale_input
        self.rescale_timesteps = False
        self.temp = annt_weight_temp  # softmax temperature for weighting the pseudo annotations
        if self.pred_noise:
            self.model_mean_type = ModelMeanType.EPSILON
        else:
            self.model_mean_type = ModelMeanType.START_X
        
        # Use float64 for accuracy.
        betas = get_named_beta_schedule(schedule, steps)
        betas = np.array(betas, dtype=np.float64)
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])
        self.denoise_steps = denoise_steps
        print(f'num_steps: {self.num_timesteps}')
        print(f"schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        alphas_cumprod_next = np.append(alphas_cumprod[1:], 0.0)
        assert alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
        log_one_minus_alphas_cumprod = np.log(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        posterior_log_variance_clipped = np.log(
            np.append(posterior_variance[1], posterior_variance[1:])   # NOTE: this is different from stable diffusion implementation
        )
        posterior_mean_coef1 = (
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - alphas_cumprod)
        )
        
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod',to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('alphas_cumprod_next',to_torch(alphas_cumprod_next))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(sqrt_alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(sqrt_one_minus_alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(log_one_minus_alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(sqrt_recip_alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(sqrt_recipm1_alphas_cumprod))
        self.register_buffer('posterior_variance',to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped',to_torch(posterior_log_variance_clipped))
        self.register_buffer('posterior_mean_coef1', to_torch(posterior_mean_coef1))
        self.register_buffer('posterior_mean_coef2', to_torch(posterior_mean_coef2))
        
        self.denoise = True if self.denoise_steps>0 else False 
        # for ddim
        self.ddim_eta = ddim_eta   #  ori code: 1.
        
        #self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method="uniform", num_ddim_timesteps=inference_steps,
                                                  #num_ddpm_timesteps=num_total_steps,verbose=True)  # note: change the max steps to denoise steps in denoising
        #ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod,
                                                                                   #ddim_timesteps=self.ddim_timesteps,
                                                                                   #eta=ddim_eta,verbose=True)
        #self.register_buffer('ddim_sigmas', to_torch(ddim_sigmas))
        #self.register_buffer('ddim_alphas',to_torch(ddim_alphas))
        #self.register_buffer('ddim_alphas_prev', to_torch(ddim_alphas_prev))
        #self.register_buffer('ddim_sqrt_one_minus_alphas', to_torch(np.sqrt(1. - ddim_alphas)))
        #sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            #(1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        #1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        #self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        if self.scale_input:
            x_start = (x_start * 2. - 1.) * self.scale 
        
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        eps = sqrt_one_minus_alphas_cumprod_t * noise
        x = sqrt_alphas_cumprod_t * x_start + eps
        
        if self.scale_input:
            x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
            x = ((x / self.scale) + 1) / 2.
        
        return x, eps

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, context=None, clip_denoised=True, denoised_fn=None, model_kwargs=None, isfirst=False
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), context_list=context, **model_kwargs)
        
        model_variance, model_log_variance = {
            # for fixedlarge, we set the initial (log-)variance like so
            # to get a better decoder log likelihood.
            ModelVarType.FIXED_LARGE: (
                np.append(self.posterior_variance[1].cpu().numpy(), self.betas[1:].cpu().numpy()),
                np.log(np.append(self.posterior_variance[1].cpu().numpy(), self.betas[1:].cpu().numpy())),
            ),
            ModelVarType.FIXED_SMALL: (
                self.posterior_variance.cpu().numpy(),
                self.posterior_log_variance_clipped.cpu().numpy(),
            ),
        }[self.model_var_type]
        model_variance, model_log_variance = to_torch(model_variance, device=x.device), to_torch(model_log_variance, device=x.device)
        model_variance = extract(model_variance, t-1, x.shape)
        model_log_variance = extract(model_log_variance, t-1, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t-1, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            
            if self.scale_input:
                x = (x * 2. - 1.) * self.scale  # convert to [-scale, scale]
                if isfirst:
                    x = (x * 2. - 1.) * self.scale
            
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
                pred_xstart_for_sample = pred_xstart.clone()
                if self.scale_input:
                   pred_xstart_for_sample = (pred_xstart_for_sample * 2. - 1.) * self.scale
            else:
                pred_xstart_for_sample = self._predict_xstart_from_eps(x_t=x, t=t-1, eps=model_output)
                pred_xstart = pred_xstart_for_sample.clone()
                if self.scale_input:
                    pred_xstart = torch.clamp(pred_xstart_for_sample, min=-1 * self.scale, max=self.scale)
                    pred_xstart = ((pred_xstart / self.scale) + 1) / 2.

                
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart_for_sample, x_t=x, t=t-1
            )
        else:
            raise NotImplementedError(self.model_mean_type)
        assert (
            len(model_mean.shape) == len(model_log_variance.shape) == len(pred_xstart.shape) == len(x.shape)
        )
        
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def p_sample(
        self,
        model,
        x,
        t,
        time_next,
        context=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        isfirst=False
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            context=context,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            isfirst=isfirst
        )
        noise = torch.randn_like(x)
        nonzero_mask = (
            (time_next != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_q(
        self,
        model,
        x,
        t,
        time_next,
        context=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        pred_xstart = model(x, self._scale_timesteps(t), context_list=context)
        nonzero_mask = (
            (time_next != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample_next, eps = self.q_sample(pred_xstart, time_next)
        return {"sample": sample_next, "pred_xstart": pred_xstart}
    
    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        context=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        inference_steps = -1,
        start_timesteps=-1,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            #device = next(model.parameters()).device
            device = context[0].device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
            
        all_step_out, all_step_in = [],[]
        total_timesteps, sampling_timesteps = self.num_timesteps, inference_steps
        start_timesteps = start_timesteps if start_timesteps>0 else total_timesteps  # custom start steps
        
        times = torch.linspace(0, start_timesteps, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        plot_steps = min(sampling_timesteps, 10)
        interv = sampling_timesteps // plot_steps
        sample_indices = times[:-1][::interv]
        #sample_indices = times[:-1]
        assert len(sample_indices) == plot_steps
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for idx, (time, time_next) in enumerate(time_pairs):
            t = torch.tensor([time] * shape[0], device=device)
            t_next = torch.tensor([time_next] * shape[0], device=device)
            if self.scale_input:
                img = torch.clamp(img, min=-1 * self.scale, max=self.scale)   
                img = ((img / self.scale) + 1) / 2
            if time in sample_indices:
                all_step_in.append(img.squeeze(1).cpu())
            isfirst=True if idx==0 else False
            with torch.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    t_next,
                    context=context,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    isfirst=isfirst
                )
                if time in sample_indices:
                    all_step_out.append(out["pred_xstart"].squeeze(1))
                if time_next <= 0:
                    all_step_out = torch.stack(all_step_out).transpose(0,1)         
                    all_step_in = torch.stack(all_step_in).transpose(0,1) 
                    return out["pred_xstart"], (all_step_out, all_step_in)
                
                img = out["sample"]
                # added
                #if self.scale_input:
                    #img = torch.clamp(img, min=-1 * self.scale, max=self.scale)   
                    #img = ((img / self.scale) + 1) / 2 
                

    #@torch.no_grad()
    def ddim_sample_new(self,
               model,
               inference_steps,
               denoise_steps,
               shape,
               context=None,
               callback=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               annt_gen=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
   
        # sampling
        size = shape
        num_total_steps = self.num_timesteps if denoise_steps<=0 else denoise_steps 
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method="uniform", num_ddim_timesteps=inference_steps,
                                                  num_ddpm_timesteps=num_total_steps,verbose=False)  # note: change the max steps to denoise steps in denoising 
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=self.alphas_cumprod.cpu().numpy(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=self.ddim_eta,verbose=False)
        self.ddim_sigmas =  to_torch(ddim_sigmas).cuda()
        self.ddim_alphas = to_torch(ddim_alphas).cuda()
        self.ddim_alphas_prev =  to_torch(ddim_alphas_prev).cuda()
        self.ddim_sqrt_one_minus_alphas =  to_torch(np.sqrt(1. - ddim_alphas)).cuda()
        sigmas_for_original_sampling_steps = self.ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.ddim_sigmas_for_original_num_steps = sigmas_for_original_sampling_steps.cuda()

        samples = self.ddim_sampling(model,
                                    size,
                                    context=context,
                                    callback=callback,
                                    img_callback=img_callback,
                                    quantize_denoised=quantize_x0,
                                    mask=mask, x0=x0,
                                    ddim_use_original_steps=False,
                                    noise_dropout=noise_dropout,
                                    temperature=temperature,
                                    score_corrector=score_corrector,
                                    corrector_kwargs=corrector_kwargs,
                                    x_T=x_T,
                                    annt_gen = annt_gen,
                                    log_every_t=log_every_t,
                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                    unconditional_conditioning=unconditional_conditioning,
                                    )
        return samples
    
    
    #@torch.no_grad()
    def ddim_sampling(self, model, shape,
                      context=None,
                      x_T=None, annt_gen=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        device = context[0].device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        #print(f"Running DDIM Sampling with {total_steps} timesteps")

        all_step_out, all_step_in = [],[]
        for i, step in enumerate(time_range):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            islast=i==len(timesteps)-1
            outs = self.p_sample_ddim(model, img, ts, index=index, context=context, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      all_step_out=all_step_out,
                                      all_step_in=all_step_in,
                                      annt_gen=annt_gen,
                                      islast=islast)
            img, pred_x0 = outs

        #all_step_out = torch.stack(all_step_out).transpose(0,1)         
        #all_step_in = torch.stack(all_step_in).transpose(0,1)  
        
        #return pred_x0, (all_step_out, all_step_in)
        return pred_x0
            
    #@torch.no_grad()
    def p_sample_ddim(self, model, x, t, index, context=None, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, annt_gen=None, all_step_out=None, all_step_in=None, islast=False):
        b, *_, device = *x.shape, x.device
        
        alphas = self.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)
        
        #all_step_in.append(x.squeeze(1).cpu())
        
        if self.scale_input:
            # normalize input
            x = torch.clamp(x, min=-1 * self.scale, max=self.scale)   
            x = ((x / self.scale) + 1) / 2 
        
        pred_x0 = model(x, t, context_list=context)
        
        if self.scale_input:
            x = (x * 2. - 1.) * self.scale
            pred_x0_for_sample = (pred_x0 * 2. - 1.) * self.scale
                            
        e_t = self._predict_eps_from_xstart(x, t, pred_x0)
        # current prediction for x_0
        #pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        #x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        x_prev = a_prev.sqrt() * pred_x0_for_sample + dir_xt + noise
        
        if self.scale_input:
            x_prev = torch.clamp(x_prev, min=-1 * self.scale, max=self.scale)   
            x_prev = ((x_prev / self.scale) + 1) / 2 
        
        #all_step_out.append(pred_x0.squeeze(1).cpu())
        
        return x_prev, pred_x0
    


    
