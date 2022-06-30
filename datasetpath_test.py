import numpy as np
import matplotlib.pyplot as plt
a = np.load('datasetpath/2_x2.npy')
b = np.load('datasetpath/2_label.npy')
print(a.shape, b.shape)
plt.imsave('dataset_test/a.png', a[0, :, 4, :])
plt.imsave('dataset_test/b.png', b[0, :, 4, :])
# seems ok