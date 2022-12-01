import numpy
import matplotlib.pyplot as plt
import torch
image = torch.load('beadslikedata/0010_x1.pt').detach().numpy()
image = image[0, :, :, 200]
plt.imsave('data_vis/0000_x1.png', image, format='png', dpi=250, cmap='gray', vmax=1.0, vmin=0.0)