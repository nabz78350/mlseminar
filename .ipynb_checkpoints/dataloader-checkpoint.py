import torch

class TimeSeriesDataset(torch.utils.data.Dataset):   
    def __init__(self, X, seq_len=100):
        self.X = X
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len-1)

    def __getitem__(self, index):
        return (self.X[index:index+self.seq_len])