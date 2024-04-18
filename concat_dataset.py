import os
import numpy as np
import tifffile

path = "_var_num_realisticdata6"
use_numpy = True
if use_numpy:
    load = np.load
    save = np.save
    ext  = "npy"
else:
    load = tifffile.imread
    save = tifffile.imwrite
    ext  = "tif"

sample = load(os.path.join(path, f"0000_labelz.{ext}"))
dir = os.listdir(path)
c, z, x, y = sample.shape

train_image = np.zeros((c, z, x, y*20))
c = 0
for name in (dir):
    if "z" in name:
        image = load(os.path.join(path, name))
        train_image[:, :, :, c:c+y] += image
        c+=y
if use_numpy:
    train_image = train_image * (2 ** 16 - 1)
os.makedirs(path+"_batch", exist_ok=True)
tifffile.imwrite(os.path.join(path+"_batch", f"train8000_val2000.tif"),
     train_image.astype(np.uint16))