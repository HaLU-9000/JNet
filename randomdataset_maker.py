import numpy as np
import time
import torch
from makedata   import make_data
torch.manual_seed(617)
np.random.seed(617)

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

for i in range(0, 20):
    t1 = time.time()
    inp     = make_data(8640, (768, 768, 768))
    t2 = time.time()
    print(f'{t2 - t1} s')
    np.save(f'microglialikedataset/{str(i).zfill(4)}_label.npy', inp)