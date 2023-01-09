import torch
arr = torch.load("beadsscore2/001_score.pt")
arr = (arr - 0.5) * 2
arr = torch.save(arr, "beadsscore3/001_score.pt")
#print(arr.min(), arr.max())
#arr = torch.load("beadsscore3/001_score.pt")
#print(arr.min(), arr.max())
