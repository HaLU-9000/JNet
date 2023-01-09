import torch
arr = torch.load("beadsdata/Beads2um_920_LP5_Ch2_HV95_offfset-15_x25_average4_001.pt")
arr = torch.clip(arr, min=0.1, max=1)
arr = torch.save(arr, "beadsdata2/Beads2um_920_LP5_Ch2_HV95_offfset-15_x25_average4_001.pt")