import torch
import torch
for s in (1, 2, 4, 8, 12):
    datasets = []
    for _ in range(6):
        dataset = torch.load(f'dataset2/dataset128_x{s}_{_ + 1}.pt')
        datasets.append(dataset)
        del(dataset)
    condata = torch.utils.data.ConcatDataset(datasets)
    torch.save(condata, f'dataset2/dataset128_x{s}.pt')
    del(condata)