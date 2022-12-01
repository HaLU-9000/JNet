import numpy as np
import time
import torch
from makedata   import make_beads_data
torch.manual_seed(617)
np.random.seed(617)

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

for i in range(0, 20):
    t1 = time.time()
    inp     = make_beads_data(480, (1200, 500, 500))
    t2 = time.time()
    print(f'{t2 - t1} s')
    np.save(f'beadslikedataset/{str(i).zfill(4)}_label.npy', inp)