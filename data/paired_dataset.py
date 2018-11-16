import torch
from .base_dataset import BaseDataset
from torchvision import transforms
from PIL import Image


class PairedDataset(BaseDataset):
    """
    PairedDataset for paired data, may be multiple output and not only two images
    Params:
        flist_paths(list) : A list contain the file path which contained paired dataset file list
        resize_shape(tuple): The shape of the final image (default:(256,256))
        transforms_oprs(list) : Determine which transformation used on the imgae (default:['random_crop', 'to_tensor'])
    Return:
        img1, img2...
    """
    def __init__(self, flist_paths, resize_shape=(256, 256), transforms_oprs=['random_crop', 'to_tensor']):
        self.flists = [ ]
        for flist_path in flist_paths:
            with open(flist_path, 'r') as f:
                self.flists.append(f.read().splitlines())
        self.resize_shape = resize_shape
        self.transform_initialize(resize_shape, transforms_oprs)


    def __len__(self):
        return len(self.flists[0])

    def __getitem__(self, index):
        return *[self.transforms_fun(read_img(flist[index])) for flist in self.flists]
