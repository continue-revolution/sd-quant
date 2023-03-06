from utils.sd_samplers_compvis import samplers_data_compvis
from utils.sd_samplers_kdiffusion import samplers_data_k_diffusion

all_samplers = [
    *samplers_data_k_diffusion,
    *samplers_data_compvis,
]
all_samplers_map = {x.name: x for x in all_samplers}


def create_sampler(name, model):
    if name is not None:
        config = all_samplers_map.get(name, None)
    else:
        config = all_samplers[0]

    assert config is not None, f'bad sampler name: {name}'

    sampler = config.constructor(model)
    sampler.config = config

    return sampler
