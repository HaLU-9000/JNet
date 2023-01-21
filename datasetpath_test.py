from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
folderpath = 'beadslikedata4'
a = list(sorted(Path(folderpath).glob(f'*.pt')))
for e, i in enumerate(a):
    print(i)
    if e is not 2:
        n = torch.load(i)
        print(n.shape)