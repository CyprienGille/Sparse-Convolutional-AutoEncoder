from PIL import Image
import os
import numpy as np
from tqdm import tqdm

originals_dir = "../../dataset/Flickr_resized/"
tiles_dir = "../../dataset/Flickr_tiles/"

tile_pixel_w = 128
tile_pixel_h = 128

for i, filename in enumerate(tqdm(os.listdir(originals_dir))):
    with Image.open(originals_dir + filename) as original_img:
        im = np.array(original_img)
        tiles = [
            im[x : x + tile_pixel_h, y : y + tile_pixel_w]
            for x in range(0, im.shape[0], tile_pixel_h)
            for y in range(0, im.shape[1], tile_pixel_w)
        ]

        for patch_id, patch in enumerate(tiles):
            patch_img = Image.fromarray(patch)
            patch_img.save(tiles_dir + f"image_{i}_patch_{patch_id}.png")
