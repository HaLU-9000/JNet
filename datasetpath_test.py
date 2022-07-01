import numpy as np
import matplotlib.pyplot as plt
import torch
label = torch.load('/home/haruhiko/Documents/JNet/dataset/make_data_1e7.pt').float()
print(label.shape[1]*label.shape[2]*label.shape[3] // 128**3)