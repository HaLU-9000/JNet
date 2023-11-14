from pathlib import Path
import numpy as np
import torch
from utils import save_dataset, save_label

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Building data on device {device}.")
folderpath    = '_var_num_beadsdataset3'
outfolderpath = '_var_num_beadsdata3'
labelname     = '_label'
outlabelname  = '_label'
save_label(folderpath, outfolderpath, labelname, outlabelname)