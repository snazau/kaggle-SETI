import torch
from sklearn.model_selection import StratifiedKFold

import dataset


class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            y = y.numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        n_batches = int(len(y) / batch_size)
        self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle)
        self.X = torch.randn(len(y),1).numpy()
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0, int(1e8), size=()).item()
        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return len(self.y)


if __name__ == "__main__":
    import numpy as np
    import __main__
    print("Run of", __main__.__file__)

    from torch.utils.data import TensorDataset, DataLoader

    X = torch.randn(100, 20)
    y = torch.randint(0, 1, size=(100,))

    for i in range(90):
        y[i] = 1

    for i in range(10):
        y[90 + i] = 0

    print(np.unique(y.numpy(), return_counts=True))

    data_loader = DataLoader(
        dataset=TensorDataset(X, y),
        batch_sampler=StratifiedBatchSampler(y, batch_size=8)
    )

    for index, sample in enumerate(data_loader):
        print("index", index)
        print(sample[-1])
        # exit()
