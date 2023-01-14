"""Small script to resize all Flickr images to 2944x3584 
for use by the CAE (23x28 128x128 patches).
This script also renames the images.
"""
#%%
from PIL import Image
import os
from tqdm import tqdm

dataset_dir = "../../dataset/flickr/original/"
output_dir = "../../dataset/Flickr_small/"

#%%
def get_dims_for_crop_to_ratio(w_init, h_init, target_ratio=5 / 3):
    """Compute the amount of pixels to shave off the width of an image
    to obtain the desired aspect ratio. (return format: (n_width, 0))
    If that is not possible (i.e. when height> weight), 
    returns the number of pixels to crop along the height instead (return format: (0, n_height))"""
    n = int(w_init - target_ratio * h_init)
    if n >= 0:
        return (n, 0)
    return (0, int(h_init - (1 / target_ratio) * w_init))


#%%
# Note :
# Our current Flickr dataset contains images of multiple sizes:
# [(3840, 5760, 3), (3744, 5616, 3), (5616, 3744, 3), (3681, 5521, 3),
# (4096, 6144, 3), (3010, 3664, 3), (5472, 3648, 3), (3648, 5472, 3),
# (3768, 5652, 3), (3807, 5710, 3), (3836, 5529, 3), (3775, 5663, 3)]
# We resize to multiples of 128 to avoid padding

target_width, target_height = 1280, 768
target_ratio = target_width / target_height
for i, filename in enumerate(tqdm(os.listdir(dataset_dir))):
    with Image.open(dataset_dir + filename) as img:
        w, h = img.size
        diff_w, diff_h = get_dims_for_crop_to_ratio(w, h, target_ratio)
        # diff_w, diff_h = w - target_width, h - target_height
        left, upper, right, lower = (
            diff_w // 2,
            diff_h // 2,
            w - diff_w // 2,
            h - diff_h // 2,
        )
        cropped = img.crop((left, upper, right, lower))

        # adjust to final size
        # also takes care of rounding errors
        resized = cropped.resize((target_width, target_height))
        resized.save(output_dir + f"image_{i}.png")

# %%
