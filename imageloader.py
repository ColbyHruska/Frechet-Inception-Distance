from PIL import Image
import numpy as np
import os

class OutOfImages(Exception):
    pass

path = ""
files = []
def set_path(_path):
    path = _path
    files = os.listdir(path)
    files = [file for file in files if file[-4:] == ".png"]

def get_batch(start, n_images):
    imgs = []
    if start + n_images >= len(files):
        raise OutOfImages()
    for file in files[start:start+n_images]:
        with Image.open(os.path.join(path, file)) as img:
            imgs.append(np.array(img))
    return np.array(imgs)