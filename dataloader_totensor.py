import numpy as np
import torch

inp = np.load('dataset/make_data_1e7.npy')
inp = torch.from_numpy(inp)
torch.save(inp,f'dataset/make_data_1e7.pt')
