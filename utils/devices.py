import contextlib
import torch


def extract_device_id(args, name):
    for x in range(len(args)):
        if name in args[x]:
            return args[x + 1]

    return None


def get_cuda_device_string():
    return "cuda"


def get_optimal_device_name():
    # if torch.cuda.is_available():
    #     return get_cuda_device_string()
    # return "cpu"
    return "cuda"


def get_optimal_device():
    return torch.device(get_optimal_device_name())


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(get_cuda_device_string()):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def enable_tf32():
    if torch.cuda.is_available():

        # enabling benchmark option seems to enable a range of cards to do fp16 when they otherwise can't
        # see https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/4407
        if any([torch.cuda.get_device_capability(devid) == (7, 5) for devid in range(0, torch.cuda.device_count())]):
            torch.backends.cudnn.benchmark = True

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True




cpu = torch.device("cpu")
device = device_interrogate = device_gfpgan = device_esrgan = device_codeformer = None
dtype = torch.float16
dtype_vae = torch.float32
dtype_unet = torch.float16
unet_needs_upcast = False


def cond_cast_unet(input):
    return input.to(dtype_unet) if unet_needs_upcast else input


def cond_cast_float(input):
    return input.float() if unet_needs_upcast else input


def randn(seed, shape):
    torch.manual_seed(seed)
    return torch.randn(shape, device=device)


def randn_without_seed(shape):
    if device.type == 'mps':
        return torch.randn(shape, device=cpu).to(device)
    return torch.randn(shape, device=device)


def autocast(disable=False):

    if disable:
        return contextlib.nullcontext()

    if dtype == torch.float32:
        return contextlib.nullcontext()

    return torch.autocast("cuda")


def without_autocast(disable=False):
    return torch.autocast("cuda", enabled=False) if torch.is_autocast_enabled() and not disable else contextlib.nullcontext()


