import numpy as np
import time
import torch
from makedata   import make_beads_data
torch.manual_seed(617)
np.random.seed(617)

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

dataset_name = "_var_num_beadsdataset"

for i in range(0, 16):
    t1 = time.time()
    inp     = make_beads_data(9600 - i * 300, (1200, 500, 500))
    t2 = time.time()
    print(f'{t2 - t1} s')
    np.save(f'{dataset_name}/{str(i).zfill(4)}_label.npy', inp)


# testdata
for i in range(0, 4):
    t1 = time.time()
    inp     = make_beads_data(9600 - (i + 6) * 300, (1200, 500, 500))
    t2 = time.time()
    print(f'{t2 - t1} s')
    np.save(f'{dataset_name}/{str(i+16).zfill(4)}_label.npy', inp)