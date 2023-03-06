import sys

import torch
import numpy as np
from PIL import Image
import random
from typing import Any, Dict, List

from ldm.data.util import AddMiDaS
from ldm.models.diffusion.ddpm import LatentDepth2ImageDiffusion

from einops import repeat, rearrange

import utils.devices as devices
import utils.prompt_parser as prompt_parser
from utils.sd_samplers import create_sampler

# some of those options should not be changed at all because they would break the model, so I removed them from options.
opt_C = 4
opt_f = 8
device = "cuda"


def single_sample_to_image(sample, sd_model, approximation=None):
    x_sample = decode_first_stage(sd_model, sample.unsqueeze(0))[0]

    x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
    x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
    x_sample = x_sample.astype(np.uint8)
    return Image.fromarray(x_sample)


def sample_to_image(samples, sd_model, index=0, approximation=None):
    return single_sample_to_image(samples[index], sd_model, approximation)


def txt2img_image_conditioning(sd_model, x, width, height):
    if sd_model.model.conditioning_key not in {'hybrid', 'concat'}:
        # Dummy zero conditioning if we're not using inpainting model.
        # Still takes up a bit of memory, but no encoder call.
        # Pretty sure we can just make this a 1x1 image since its not going to be used besides its batch size.
        return x.new_zeros(x.shape[0], 5, 1, 1, dtype=x.dtype, device=x.device)

    # The "masked-image" in this case will just be all zeros since the entire image is masked.
    image_conditioning = torch.zeros(
        x.shape[0], 3, height, width, device=x.device)
    image_conditioning = sd_model.get_first_stage_encoding(
        sd_model.encode_first_stage(image_conditioning))

    # Add the fake full 1s mask to the first dimension.
    image_conditioning = torch.nn.functional.pad(
        image_conditioning, (0, 0, 0, 0, 1, 0), value=1.0)
    image_conditioning = image_conditioning.to(x.dtype)

    return image_conditioning


class StableDiffusionProcessing:
    """
    The first set of paramaters: sd_models -> do_not_reload_embeddings represent the minimum required to create a StableDiffusionProcessing
    """

    def __init__(self, sd_model=None, outpath_samples=None, outpath_grids=None, prompt: str = "", styles: List[str] = None, seed: int = -1, subseed: int = -1, subseed_strength: float = 0, seed_resize_from_h: int = -1, seed_resize_from_w: int = -1, seed_enable_extras: bool = True, sampler_name: str = None, batch_size: int = 1, n_iter: int = 1, steps: int = 50, cfg_scale: float = 7.0, width: int = 512, height: int = 512, restore_faces: bool = False, tiling: bool = False, do_not_save_samples: bool = False, do_not_save_grid: bool = False, extra_generation_params: Dict[Any, Any] = None, overlay_images: Any = None, negative_prompt: str = None, eta: float = None, do_not_reload_embeddings: bool = False, denoising_strength: float = 0, ddim_discretize: str = None, s_churn: float = 0.0, s_tmax: float = None, s_tmin: float = 0.0, s_noise: float = 1.0, override_settings: Dict[str, Any] = None, override_settings_restore_afterwards: bool = True, sampler_index: int = None, script_args: list = None):
        if sampler_index is not None:
            print("sampler_index argument for StableDiffusionProcessing does not do anything; use sampler_name", file=sys.stderr)
        self.sd_model: torch.nn.Module = sd_model
        self.outpath_samples: str = outpath_samples
        self.outpath_grids: str = outpath_grids
        self.prompt: str = prompt
        self.prompt_for_display: str = None
        self.negative_prompt: str = (negative_prompt or "")
        self.styles: list = styles or []
        self.seed: int = seed
        self.subseed: int = subseed
        self.subseed_strength: float = subseed_strength
        self.seed_resize_from_h: int = seed_resize_from_h
        self.seed_resize_from_w: int = seed_resize_from_w
        self.sampler_name: str = sampler_name
        self.batch_size: int = batch_size
        self.n_iter: int = n_iter
        self.steps: int = steps
        self.cfg_scale: float = cfg_scale
        self.width: int = width
        self.height: int = height
        self.restore_faces: bool = restore_faces
        self.tiling: bool = tiling
        self.do_not_save_samples: bool = do_not_save_samples
        self.do_not_save_grid: bool = do_not_save_grid
        self.extra_generation_params: dict = extra_generation_params or {}
        self.overlay_images = overlay_images
        self.eta = eta
        self.do_not_reload_embeddings = do_not_reload_embeddings
        self.paste_to = None
        self.color_corrections = None
        self.denoising_strength: float = denoising_strength
        self.sampler_noise_scheduler_override = None
        self.ddim_discretize = ddim_discretize
        self.s_churn = s_churn
        self.s_tmin = s_tmin
        # not representable as a standard ui option
        self.s_tmax = s_tmax or float('inf')
        self.s_noise = s_noise
        self.override_settings = {k: v for k, v in (
            override_settings or {}).items()}
        self.override_settings_restore_afterwards = override_settings_restore_afterwards
        self.is_using_inpainting_conditioning = False
        self.disable_extra_networks = False

        if not seed_enable_extras:
            self.subseed = -1
            self.subseed_strength = 0
            self.seed_resize_from_h = 0
            self.seed_resize_from_w = 0

        self.scripts = None
        self.script_args = script_args
        self.all_prompts = None
        self.all_negative_prompts = None
        self.all_seeds = None
        self.all_subseeds = None
        self.iteration = 0

    def txt2img_image_conditioning(self, x, width=None, height=None):
        self.is_using_inpainting_conditioning = self.sd_model.model.conditioning_key in {
            'hybrid', 'concat'}

        return txt2img_image_conditioning(self.sd_model, x, width or self.width, height or self.height)

    def depth2img_image_conditioning(self, source_image):
        # Use the AddMiDaS helper to Format our source image to suit the MiDaS model
        transformer = AddMiDaS(model_type="dpt_hybrid")
        transformed = transformer(
            {"jpg": rearrange(source_image[0], "c h w -> h w c")})
        midas_in = torch.from_numpy(
            transformed["midas_in"][None, ...]).to(device=device)
        midas_in = repeat(midas_in, "1 ... -> n ...", n=self.batch_size)

        conditioning_image = self.sd_model.get_first_stage_encoding(
            self.sd_model.encode_first_stage(source_image))
        conditioning = torch.nn.functional.interpolate(
            self.sd_model.depth_model(midas_in),
            size=conditioning_image.shape[2:],
            mode="bicubic",
            align_corners=False,
        )

        (depth_min, depth_max) = torch.aminmax(conditioning)
        conditioning = 2. * (conditioning - depth_min) / \
            (depth_max - depth_min) - 1.
        return conditioning

    def edit_image_conditioning(self, source_image):
        conditioning_image = self.sd_model.encode_first_stage(
            source_image).mode()

        return conditioning_image

    def img2img_image_conditioning(self, source_image, latent_image, image_mask=None):
        source_image = devices.cond_cast_float(source_image)

        # HACK: Using introspection as the Depth2Image model doesn't appear to uniquely
        # identify itself with a field common to all models. The conditioning_key is also hybrid.
        if isinstance(self.sd_model, LatentDepth2ImageDiffusion):
            return self.depth2img_image_conditioning(source_image)

        if self.sd_model.cond_stage_key == "edit":
            return self.edit_image_conditioning(source_image)

        if self.sampler.conditioning_key in {'hybrid', 'concat'}:
            return self.inpainting_image_conditioning(source_image, latent_image, image_mask=image_mask)

        # Dummy zero conditioning if we're not using inpainting or depth model.
        return latent_image.new_zeros(latent_image.shape[0], 5, 1, 1)

    def init(self, all_prompts, all_seeds, all_subseeds):
        pass

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
        raise NotImplementedError()

    def close(self):
        self.sampler = None


# from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3
def slerp(val, low, high):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    dot = (low_norm*high_norm).sum(1)

    if dot.mean() > 0.9995:
        return low * val + high * (1 - val)

    omega = torch.acos(dot)
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1) * \
        low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res


def create_random_tensors(shape, seeds, subseeds=None, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0, p=None):
    eta_noise_seed_delta = 0
    xs = []

    # if we have multiple seeds, this means we are working with batch size>1; this then
    # enables the generation of additional tensors with noise that the sampler will use during its processing.
    # Using those pre-generated tensors instead of simple torch.randn allows a batch with seeds [100, 101] to
    # produce the same images as with two batches [100], [101].
    if p is not None and p.sampler is not None and (len(seeds) > 1 or eta_noise_seed_delta > 0):
        sampler_noises = [[]
                          for _ in range(p.sampler.number_of_needed_noises(p))]
    else:
        sampler_noises = None

    for i, seed in enumerate(seeds):
        noise_shape = shape if seed_resize_from_h <= 0 or seed_resize_from_w <= 0 else (
            shape[0], seed_resize_from_h//8, seed_resize_from_w//8)

        subnoise = None
        if subseeds is not None:
            subseed = 0 if i >= len(subseeds) else subseeds[i]

            subnoise = devices.randn(subseed, noise_shape)

        # randn results depend on device; gpu and cpu get different results for same seed;
        # the way I see it, it's better to do this on CPU, so that everyone gets same result;
        # but the original script had it like this, so I do not dare change it for now because
        # it will break everyone's seeds.
        noise = devices.randn(seed, noise_shape)

        if subnoise is not None:
            noise = slerp(subseed_strength, noise, subnoise)

        if noise_shape != shape:
            x = devices.randn(seed, shape)
            dx = (shape[2] - noise_shape[2]) // 2
            dy = (shape[1] - noise_shape[1]) // 2
            w = noise_shape[2] if dx >= 0 else noise_shape[2] + 2 * dx
            h = noise_shape[1] if dy >= 0 else noise_shape[1] + 2 * dy
            tx = 0 if dx < 0 else dx
            ty = 0 if dy < 0 else dy
            dx = max(-dx, 0)
            dy = max(-dy, 0)

            x[:, ty:ty+h, tx:tx+w] = noise[:, dy:dy+h, dx:dx+w]
            noise = x

        if sampler_noises is not None:
            cnt = p.sampler.number_of_needed_noises(p)

            if eta_noise_seed_delta > 0:
                torch.manual_seed(seed + eta_noise_seed_delta)

            for j in range(cnt):
                sampler_noises[j].append(
                    devices.randn_without_seed(tuple(noise_shape)))

        xs.append(noise)

    if sampler_noises is not None:
        p.sampler.sampler_noises = [torch.stack(
            n).to(device) for n in sampler_noises]

    x = torch.stack(xs).to(device)
    return x


def decode_first_stage(model, x):
    with devices.autocast(disable=x.dtype == devices.dtype_vae):
        x = model.decode_first_stage(x)

    return x


def get_fixed_seed(seed):
    if seed is None or seed == '' or seed == -1:
        return int(random.randrange(4294967294))

    return seed


def fix_seed(p):
    p.seed = get_fixed_seed(p.seed)
    p.subseed = get_fixed_seed(p.subseed)


def quote(text):
    if ',' not in str(text):
        return text

    text = str(text)
    text = text.replace('\\', '\\\\')
    text = text.replace('"', '\\"')
    return f'"{text}"'


def create_infotext(p, all_prompts, all_seeds, all_subseeds, comments=None, iteration=0, position_in_batch=0):
    index = position_in_batch + iteration * p.batch_size

    clip_skip = 1

    generation_params = {
        "Steps": p.steps,
        "Sampler": p.sampler_name,
        "CFG scale": p.cfg_scale,
        "Image CFG scale": getattr(p, 'image_cfg_scale', None),
        "Seed": all_seeds[index],
        "Size": f"{p.width}x{p.height}",
        "Model": "Pastel-Mix",
        "Variation seed": (None if p.subseed_strength == 0 else all_subseeds[index]),
        "Variation seed strength": (None if p.subseed_strength == 0 else p.subseed_strength),
        "Seed resize from": (None if p.seed_resize_from_w == 0 or p.seed_resize_from_h == 0 else f"{p.seed_resize_from_w}x{p.seed_resize_from_h}"),
        "Denoising strength": getattr(p, 'denoising_strength', None),
        "Clip skip": None if clip_skip <= 1 else clip_skip,
    }

    generation_params.update(p.extra_generation_params)

    generation_params_text = ", ".join(
        [k if k == v else f'{k}: {quote(v)}' for k, v in generation_params.items() if v is not None])

    negative_prompt_text = "\nNegative prompt: " + \
        p.all_negative_prompts[index] if p.all_negative_prompts[index] else ""

    return f"{all_prompts[index]}{negative_prompt_text}\n{generation_params_text}".strip()


def process_images(p: StableDiffusionProcessing):
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""

    if type(p.prompt) == list:
        assert(len(p.prompt) > 0)
    else:
        assert p.prompt is not None

    devices.torch_gc()

    seed = get_fixed_seed(p.seed)
    subseed = get_fixed_seed(p.subseed)

    comments = {}
    p.all_prompts = p.batch_size * p.n_iter * [p.prompt]
    p.all_negative_prompts = p.batch_size * p.n_iter * [p.negative_prompt]

    if type(seed) == list:
        p.all_seeds = seed
    else:
        p.all_seeds = [int(seed) + (x if p.subseed_strength == 0 else 0)
                       for x in range(len(p.all_prompts))]

    if type(subseed) == list:
        p.all_subseeds = subseed
    else:
        p.all_subseeds = [int(subseed) + x for x in range(len(p.all_prompts))]

    def infotext(iteration=0, position_in_batch=0):
        return create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, comments, iteration, position_in_batch)

    output_images = []

    cached_uc = [None, None]
    cached_c = [None, None]

    def get_conds_with_caching(function, required_prompts, steps, cache):
        """
        Returns the result of calling function(shared.sd_model, required_prompts, steps)
        using a cache to store the result if the same arguments have been used before.

        cache is an array containing two elements. The first element is a tuple
        representing the previously used arguments, or None if no arguments
        have been used before. The second element is where the previously
        computed result is stored.
        """

        if cache[0] is not None and (required_prompts, steps) == cache[0]:
            return cache[1]

        with devices.autocast():
            cache[1] = function(p.sd_model, required_prompts, steps)

        cache[0] = (required_prompts, steps)
        return cache[1]

    with torch.no_grad(), p.sd_model.ema_scope():
        with devices.autocast():
            p.init(p.all_prompts, p.all_seeds, p.all_subseeds)

        for n in range(p.n_iter):
            p.iteration = n

            prompts = p.all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            negative_prompts = p.all_negative_prompts[n *
                                                      p.batch_size:(n + 1) * p.batch_size]
            seeds = p.all_seeds[n * p.batch_size:(n + 1) * p.batch_size]
            subseeds = p.all_subseeds[n * p.batch_size:(n + 1) * p.batch_size]

            if len(prompts) == 0:
                break

            uc = get_conds_with_caching(
                prompt_parser.get_learned_conditioning, negative_prompts, p.steps, cached_uc)
            c = get_conds_with_caching(
                prompt_parser.get_multicond_learned_conditioning, prompts, p.steps, cached_c)

            with devices.without_autocast() if devices.unet_needs_upcast else devices.autocast():
                samples_ddim = p.sample(conditioning=c, unconditional_conditioning=uc, seeds=seeds,
                                        subseeds=subseeds, subseed_strength=p.subseed_strength, prompts=prompts)

            x_samples_ddim = [decode_first_stage(p.sd_model, samples_ddim[i:i+1].to(
                dtype=devices.dtype_vae))[0].cpu() for i in range(samples_ddim.size(0))]

            x_samples_ddim = torch.stack(x_samples_ddim).float()
            x_samples_ddim = torch.clamp(
                (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            del samples_ddim

            devices.torch_gc()

            for i, x_sample in enumerate(x_samples_ddim):
                x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                x_sample = x_sample.astype(np.uint8)

                image = Image.fromarray(x_sample)

                image.info["parameters"] = infotext(n, i)
                output_images.append(image)

            del x_samples_ddim

            devices.torch_gc()

    devices.torch_gc()

    print("Start processing transcript...")
    batch_size = len(output_images)
    transcript = [[] * batch_size]
    for intermediate_batch in p.sampler.transcript:
        for b in range(batch_size):
            transcript[b].append(sample_to_image(
                intermediate_batch, p.sd_model, b))
    print("Finish processing transcript...")

    return output_images, transcript


class StableDiffusionProcessingTxt2Img(StableDiffusionProcessing):
    sampler = None

    def __init__(self, enable_hr: bool = False, denoising_strength: float = 0.75, firstphase_width: int = 0, firstphase_height: int = 0, hr_scale: float = 2.0, hr_upscaler: str = None, hr_second_pass_steps: int = 0, hr_resize_x: int = 0, hr_resize_y: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.enable_hr = enable_hr
        self.denoising_strength = denoising_strength
        self.hr_scale = hr_scale
        self.hr_upscaler = hr_upscaler
        self.hr_second_pass_steps = hr_second_pass_steps
        self.hr_resize_x = hr_resize_x
        self.hr_resize_y = hr_resize_y
        self.hr_upscale_to_x = hr_resize_x
        self.hr_upscale_to_y = hr_resize_y

        if firstphase_width != 0 or firstphase_height != 0:
            self.hr_upscale_to_x = self.width
            self.hr_upscale_to_y = self.height
            self.width = firstphase_width
            self.height = firstphase_height

        self.truncate_x = 0
        self.truncate_y = 0
        self.applied_old_hires_behavior_to = None

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
        self.sampler = create_sampler(
            self.sampler_name, self.sd_model)

        x = create_random_tensors([opt_C, self.height // opt_f, self.width // opt_f], seeds=seeds, subseeds=subseeds,
                                  subseed_strength=self.subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w, p=self)
        samples = self.sampler.sample(self, x, conditioning, unconditional_conditioning,
                                      image_conditioning=self.txt2img_image_conditioning(x))
        return samples
