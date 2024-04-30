import os
import tifffile
import numpy as np
import torch
import torch.nn.functional as F
base = "_angle_blur/angle00"
new  = "_angle_blur_up/angle00"

os.makedirs(new, exist_ok=True)
images = [i for i in sorted(os.listdir(base))]
for i in images:
    print(i)
    image = tifffile.imread(os.path.join(base, i))
    print(image.shape)
    image = F.interpolate(torch.tensor(image[None, None]*1.0), 
                          scale_factor=(10,1,1)).detach().numpy()
    tifffile.imwrite(os.path.join(new, i),
                     image[0,0].astype(np.uint16), dtype=np.uint16)