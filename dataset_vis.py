import numpy
import matplotlib.pyplot as plt
import torch
image = torch.load('randomdata/0000_x1.pt').detach().numpy()
image = image[0, :, 300, :]
plt.imsave('data_vis/0000_x1_xx.png', image, format='png', dpi=250, cmap='gray')