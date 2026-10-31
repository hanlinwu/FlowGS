import os
from concurrent.futures import ThreadPoolExecutor

from PIL import Image


def mkdir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_file_list(path, regx=None):
    if os.path.isdir(path):
        import re

        file_names = sorted(os.listdir(path))
        if regx is not None:
            file_names = [name for name in file_names if re.match(regx, name)]
        return [os.path.join(path, name) for name in file_names]
    with open(path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def read_image(path):
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def read_images_parallel(paths, max_workers=8):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(read_image, paths))
