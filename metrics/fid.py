import torch
from pytorch_fid import fid_score


def calc_fid(paths, batch_size=16, device=None, dims=2048):
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif isinstance(device, torch.device):
        device = str(device)
    return fid_score.calculate_fid_given_paths(paths, batch_size, device, dims, num_workers=8)
