import torch
arr = torch.load("beadsdata/Beads2um_920_LP5_Ch2_HV95_offfset-15_x25_average4_001.pt")
minimum = 0.1
arr     = torch.clip(arr, min=minimum, max=1)
arr     = (arr - minimum) * (1 / (1 - minimum))
print(arr.min(), arr.max())
torch.save(arr, "beadsdata3/Beads2um_920_LP5_Ch2_HV95_offfset-15_x25_average4_001.pt")

#arr = torch.load("beadsdata2/Beads2um_920_LP5_Ch2_HV95_offfset-15_x25_average4_001.pt")
#arr = (arr - 0.1) * (1.0 / 0.9)
#

