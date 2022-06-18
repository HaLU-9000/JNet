import numpy as np
import time
import torch
from makedata   import make_data
from dataloader import Blur, CustomDataset
torch.manual_seed(617)
np.random.seed(617)

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")


inp = np.load('dataset/make_data_1e7.npy')
inp = torch.from_numpy(inp, )
inp = inp[:, :1536, ...]
for _ in (1, 2, 4, 8, 12):
    dataset = CustomDataset(inp, _, 128, 128, 128, 141, 7, 7, bet_xy=6, bet_z=35)
    print(f'CustomDataset(x{str(_)}) done!!')
    torch.save(dataset,f'dataset/dataset128_x{str(_)}.pt')
    del(dataset)
