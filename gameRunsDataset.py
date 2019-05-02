import h5py
import torch.utils.data as data
from numpy import array, double
from torchvision import transforms
from torch import from_numpy, DoubleTensor

class GameRuns(data.Dataset):
    #base_folder = '/home/phmignot/storage/data/'
    #base_folder = './data/'
    #filename = 'concatF4C2G3K4.hdf5'

    def __init__(self, folder, filename, imgs_key='runs', transform=None):
        self.hdf5file = folder + filename
        self.imgs_key = imgs_key
        self.transform = transform

    def __len__(self):
        with h5py.File(self.hdf5file, 'r') as db:
            lens=len(db[self.imgs_key])
        return lens

    def __getitem__(self, idx):
        with h5py.File(self.hdf5file,'r') as db:
            image = (db[self.imgs_key][idx])
            label = array([[[0]]])
        # if self.transform:
        #     image = self.transform(image)
        #     label = transforms.ToTensor()(label)
        image = transforms.ToTensor()(image)
        image = image.float()
        image = transforms.functional.normalize(image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        label = transforms.ToTensor()(label)
        return image.float(), label
