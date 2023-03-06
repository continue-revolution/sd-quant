from omegaconf import OmegaConf
import safetensors.torch
from utils.sd_disable_initialization import DisableInitialization
from ldm.util import instantiate_from_config


chckpoint_dict_replacements = {
    'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
    'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
    'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
}


def transform_checkpoint_dict_key(k):
    for text, replacement in chckpoint_dict_replacements.items():
        if k.startswith(text):
            k = replacement + k[len(text):]
    return k


def get_state_dict_from_checkpoint(pl_sd):
    pl_sd = pl_sd.pop("state_dict", pl_sd)
    pl_sd.pop("state_dict", None)
    sd = {}
    for k, v in pl_sd.items():
        new_key = transform_checkpoint_dict_key(k)
        if new_key is not None:
            sd[new_key] = v
    pl_sd.clear()
    pl_sd.update(sd)
    return pl_sd

def load_model(param_file: str, conf_file: str, device: str):
    sd_config = OmegaConf.load(conf_file)
    pl_sd = safetensors.torch.load_file(param_file, device=device)
    state_dict = get_state_dict_from_checkpoint(pl_sd)
    
    with DisableInitialization(disable_clip=True):
        sd_model = instantiate_from_config(sd_config.model)
    sd_model.load_state_dict(state_dict, strict=False)
    del state_dict
    
    vae = sd_model.first_stage_model
    sd_model.first_stage_model = None
    sd_model.half()
    sd_model.first_stage_model = vae
    
    sd_model.to('cuda')
    sd_model.eval()
    return sd_model
