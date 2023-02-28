import numpy
import matplotlib.pyplot as plt
import torch
image = torch.load('spinelikedata0/0010_x6.pt').detach().numpy()
print('max', image.max(), 'min', image.min())
image = image[0, 10:40, 0:100, 200]
plt.figure(figsize=(5,10))
plt.imshow(image, cmap='gray',aspect=6, vmax=1.0, vmin=0.0)
plt.savefig('data_vis/spine0010_x6.png', dpi=250, format='png')