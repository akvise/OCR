import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
import os
import config


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class TextDataset(Dataset):
    def __init__(self, data_dir, image_names, test_mode=False):
        self.data_dir = data_dir
        self.image_names = image_names
        self.test_mode = test_mode

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_path = os.path.join(self.data_dir, image_name)
        image = Image.open(image_path).convert('L')
        if self.test_mode:
            im_k = 32 / image.size[1]

            image = image.resize((int(image.size[0] * im_k), 32), resample=Image.BILINEAR)  # resize for model

        else:
            image = image.resize((600, 32), resample=Image.BILINEAR)  # resize for model

        text = image_name[:-len('.png')]
        text = [config.CHAR2LABEL[i] for i in text]
        image = self.transform(image)

        text = torch.LongTensor(text)

        return image, text

    def transform(self, image):
        transform_ops = transforms.Compose([
            transforms.ToTensor(),
            #AddGaussianNoise(0., .08 if self.test_mode else 0.)
            # transforms.Normalize()
        ])
        return transform_ops(image)