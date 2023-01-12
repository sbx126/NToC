from torchvision import datasets
from torchvision.transforms import ToTensor
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import random


class CustomMNIST(Dataset):
    def __init__(self, train, included_classes = [i for i in range(10)], num_samples = None, balance_class = False, root = "../../data"):
        self.dataset = datasets.MNIST(
            root = root,
            train = train,
            transform = ToTensor(),
            download = True
        )
        self.num_samples = num_samples
        self.balance_class = balance_class

        self.valid_idxs = []
        for i in range(len(self.dataset)):
            if self.dataset.targets[i] in included_classes:
                self.valid_idxs.append(i)

        if num_samples is not None:
            random.shuffle(self.valid_idxs)
            if not balance_class:
                self.valid_idxs = self.valid_idxs[:num_samples]
            else:
                new_valid_idxs = []
                num_samples_per_class = [0 for _ in range(10)]
                for idx in range(len(self.valid_idxs)):
                    data_idx = self.valid_idxs[idx]
                    if num_samples_per_class[self.dataset.targets[data_idx]] < num_samples / len(included_classes):
                        new_valid_idxs.append(data_idx)
                        num_samples_per_class[self.dataset.targets[data_idx]] += 1
                self.valid_idxs = new_valid_idxs

    def __len__(self):
        return len(self.valid_idxs)

    def __getitem__(self, idx):
        sample_idx = self.valid_idxs[idx]
        return self.dataset.data[sample_idx, :, :].reshape(1, 28, 28) / 255.0, self.dataset.targets[sample_idx]


class CustomSampler(Sampler):
    def __init__(self, data, weighted_list = None, num_samples = None):
        self.data = data
        self.weighted_list = weighted_list
        self.num_samples = num_samples
            
    def __iter__(self):
        indices = []
        weighted =  torch.Tensor(self.weighted_list[2])
        num_labels = torch.multinomial(weighted, self.num_samples // 2, True)

        for m in range(self.num_samples // 2): 
            index_1 = random.choice(torch.where(self.data.dataset.targets[self.data.valid_idxs] == self.weighted_list[0][num_labels[m]])[0].tolist())            
            index_2 = random.choice(torch.where(self.data.dataset.targets[self.data.valid_idxs] == self.weighted_list[1][num_labels[m]])[0].tolist())
            indices.append(index_1)
            indices.append(index_2)
        #indices = torch.cat(indices, dim=0)
        return iter(indices)

    def __len__(self):
        return len(self.data)


def get_mnist_dataloader(train = True, batch_size = 32, included_classes = [i for i in range(10)], num_samples = None, 
                         balance_class = False, root = "../../data/", shuffle = True):
    dataset = CustomMNIST(train = train, included_classes = included_classes, num_samples = num_samples, 
                          balance_class = balance_class, root = root)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size = batch_size, shuffle = shuffle, num_workers = 0
    )
    
    return data_loader


def get_weighted_dataloader(train = True, batch_size = 32, included_classes = [i for i in range(10)], num_samples = None, 
                         balance_class = False, root = "../../data/", weighted_list = None):
    mydataset = CustomMNIST(train = train, included_classes = included_classes, num_samples = num_samples, 
                          balance_class = balance_class, root = root)
    
    data_sampler = CustomSampler(mydataset, weighted_list = weighted_list, num_samples = num_samples)
    
    data_loader = torch.utils.data.DataLoader(
       dataset = mydataset, batch_size = batch_size, num_workers = 0, sampler = data_sampler
    )
    
    return data_loader
