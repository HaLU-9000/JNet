import os
import numpy as np
import tifffile

path = "/home/haruhiko/Documents/simulation_ad2/simulation_data/haystack/10000-lines-blurred-by-gaublur-std-4/blurred"
use_numpy = False
if use_numpy:
    load = np.load
    save = np.save
    ext  = "npy"
else:
    load = tifffile.imread
    save = tifffile.imwrite
    ext  = "tif"

sample = load(os.path.join(path, f"GT_45_0.{ext}"))
dir = os.listdir(path)
z, x, y = sample.shape

train_image = np.zeros((1, z, x, y*3))
c = 0
for name in (sorted(dir)):
    if "GT_90_" in name:
        image = load(os.path.join(path, name))
        train_image[:, :, :, c:c+y] += image
        c+=y
if use_numpy:
    train_image = train_image * (2 ** 16 - 1)
os.makedirs(path+"_batch", exist_ok=True)
tifffile.imwrite(os.path.join(path, f"GT_090.tif"),
     train_image.astype(np.uint16))