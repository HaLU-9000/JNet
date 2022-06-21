import numpy as np
import time
import torch
from dataset.makedata   import make_data
torch.manual_seed(617)
np.random.seed(617)

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

t1 = time.time()
inp     = make_data(1327104, (1920, 1536, 1536))
t2 = time.time()
print(f'{t2 - t1} s', )
np.save('dataset/make_data_1e7.npy', inp)
print('make_data done!!')