import os
import numpy as np
import tifffile

path = "/home/haruhiko/Documents/simulation_ad2/blurred_without_noise"
angle = "45"
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
    if f"GT_{angle}_" in name:
        image = load(os.path.join(path, name))
        train_image[:, :, :, c:c+y] += image
        c+=y
if use_numpy:
    train_image = train_image * (2 ** 16 - 1)
#os.makedirs(path+"_batch", exist_ok=True)
tifffile.imwrite(os.path.join(path, f"GT_0{angle}.tif"),
     train_image.astype(np.uint16))