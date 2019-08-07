import torch


def idx2onehot(idx, n):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)

    onehot = torch.zeros(idx.size(0), n)
    onehot = onehot.to(device)
    idx = idx.to(device)
    onehot.scatter_(1, idx, 1)

    return onehot
