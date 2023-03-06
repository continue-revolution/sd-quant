from PIL import PngImagePlugin
from torch.nn import Module
from typing import List
from utils.processing import StableDiffusionProcessingTxt2Img, process_images
from matplotlib import pyplot as plt
from matplotlib import image
from io import BytesIO
import os

def encode_pil(image):
    with BytesIO() as output_bytes:
        use_metadata = False
        metadata = PngImagePlugin.PngInfo()
        for key, value in image.info.items():
            if isinstance(key, str) and isinstance(value, str):
                metadata.add_text(key, value)
                use_metadata = True
        image.save(output_bytes, format="PNG", pnginfo=(
            metadata if use_metadata else None), quality=80)
        bytes_data = output_bytes.getvalue()
    return bytes_data

def get_sample(sd_model: Module, command: dict):
    p = StableDiffusionProcessingTxt2Img(sd_model=sd_model, **command)
    processed = process_images(p)
    images = list(map(encode_pil, processed[0]))
    transcript = [list(map(encode_pil, t)) for t in processed[1]]
    return images, transcript

def save_img(image_input):
    save_path = "outputs/"
    from pathlib import Path
    Path(save_path).mkdir(exist_ok=True)
    num_files = len([name for name in os.listdir(save_path)])
    img_filename = f'{save_path}/{num_files}.png'
    with open(img_filename, "wb") as fh:
        fh.write(image_input)
    print("saved to", img_filename)
    
def plot_transcript(tr: List[str]):
    fig = plt.figure(figsize=(256, 256))
    for i, t in enumerate(tr):
        fig.add_subplot(50, 1, i + 1)
        tmp = image.imread(BytesIO(t), format='png')
        plt.imshow(tmp)
        plt.axis('off')
        plt.title(f'{i+1}')