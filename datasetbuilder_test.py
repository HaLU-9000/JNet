import numpy as np
import torch
from makedata   import make_data
from dataloader import Blur, CustomDataset
import time
torch.manual_seed(617)
np.random.seed(617)

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")
t1 = time.time()
inp     = make_data(13, (200, 200, 200))
np.save('dataset_test/make_data_1e7.npy', inp)
t2 = time.time()

print(f'{t2 - t1} s', )
time.sleep(60)
print('make_data done!!')
for _ in (1, 2, 4, 8, 12):
    dataset = CustomDataset(inp, _, 128, 128, 128, 141, 7, 7, bet_xy=6, bet_z=35)
    print(f'CustomDataset(x{str(_)}) done!!')
    torch.save(dataset,f'dataset_test/testdataset128_x{str(_)}.pt')
t3 = time.time()
print(f'{t3 - t2} s')