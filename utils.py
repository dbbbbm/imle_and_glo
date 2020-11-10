import os
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils import data
from torchvision import datasets, transforms


class GloMNIST(data.Dataset):
    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, root, train=True, transform=None, target_transform=None, latent_dim=64):
        super(GloMNIST, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(root, data_file))
        self.z = torch.randn(len(self.data), latent_dim)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, z = self.data[index], self.z[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, z

    def __len__(self):
        return len(self.data)


class Glolatent(data.Dataset):
    def __init__(self, root):
        super(Glolatent, self).__init__()
        self.z = torch.load(os.path.join(root, 'glo_zs.pth'))

    def __getitem__(self, index):
        return self.z[index]

    def __len__(self):
        return self.z.size(0)


def get_glomnist_dataloader(bs):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    ds = GloMNIST('./MNIST/processed', transform=tf)
    dl = data.DataLoader(ds, bs, True)

    return dl, ds


def get_glolatent_dataloader(bs):
    ds = Glolatent('./')
    dl = data.DataLoader(ds, bs, True)

    return dl


def find_nearest_neighbor(x, out):
    dist = F.pairwise_distance(
        x.view(x.size(0), -1).unsqueeze(2), out.view(out.size(0), -1).t().unsqueeze(0))
    idx = dist.argmin(dim=1)
    return out[idx]
