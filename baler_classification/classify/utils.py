import os
from PIL import Image

def sort_filename(path1: str, path2: str):
    """
    Return (top, bottom) where the newer filename is used as top.
    Assumes filenames contain zero-padded timestamps.
    """
    name1 = os.path.basename(path1)
    name2 = os.path.basename(path2)

    if name1 >= name2:
        return path1, path2, False
    else:
        return path2, path1, True

def combine_images(top_img: Image.Image, bottom_img: Image.Image, img_size: int):
    orig_w, orig_h = top_img.size
    
    top_crop = top_img.crop((0, orig_h // 2, orig_w, orig_h))
    top_crop = top_crop.resize((img_size, img_size), Image.Resampling.LANCZOS)
    
    bottom_crop = bottom_img.crop((0, 0, orig_w, orig_h // 2))
    bottom_crop = bottom_crop.resize((img_size, img_size), Image.Resampling.LANCZOS)
    
    combined = Image.new('RGB', (img_size, img_size * 2))
    combined.paste(top_crop, (0, 0))
    combined.paste(bottom_crop, (0, img_size))

    return combined