import numpy as np
import torch
from makedata   import make_data
from dataloader import Blur, CustomDataset

inp     = make_data(1327104, (1920, 1536, 1536))
np.save('dataset/make_data_1e7.npy', inp)
print('make_data done!!')
for _ in (1, 2, 4, 8, 12):
    dataset = CustomDataset(inp, _, 128, 128, 128, 141, 7, 7, bet_xy=6, bet_z=35)
    print(f'CustomDataset(x{str(_)}) done!!')
    torch.save(dataset,f'dataset/dataset128_x{str(_)}.pt')