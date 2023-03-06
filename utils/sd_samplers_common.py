from collections import namedtuple


SamplerData = namedtuple(
    'SamplerData', ['name', 'constructor', 'aliases', 'options'])


def setup_img2img_steps(p, steps=None):
    if steps is not None:
        requested_steps = (steps or p.steps)
        steps = int(requested_steps / min(p.denoising_strength,
                    0.999)) if p.denoising_strength > 0 else 0
        t_enc = requested_steps - 1
    else:
        steps = p.steps
        t_enc = int(min(p.denoising_strength, 0.999) * steps)

    return steps, t_enc
