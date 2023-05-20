from torch.utils.data import Subset
class CustomSubset(Subset):
    '''A custom subset class'''
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.targets = dataset.targets # 保留targets属性
        self.classes = dataset.classes # 保留classes属性
        self.train_labels = dataset.train_labels  # 保留classes属性

    def __getitem__(self, idx): #同时支持索引访问操作
        x, targets = self.dataset[self.indices[idx]]
        return x, targets

    def __len__(self): # 同时支持取长度操作
        return len(self.indices)
