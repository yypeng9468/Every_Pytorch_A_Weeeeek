# Data loading and processing tutorial
# pytorch上来就是一句解决很多机器学习的问题大多数时间都花费在准备数据上。
# 今天的tutorial主要关注于对数据进行预处理和数据增强啦，
# 这个教程的内容主要是关于Dlib的一个人脸姿态识别的内容，通过确定人脸上68个关键点进行人脸姿态识别

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import warnings
warnings.filterwarnings("ignore")

landmarks_frame = pd.read_csv("/Users/pengyuyan/Desktop/pytorch/data/faces/face_landmarks.csv")

n = 65
img_name = landmarks_frame.iloc[n,0]
landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
landmarks = landmarks.astype("float").reshape(-1,2)

print("Image name: {}".format(img_name))
print("Landmarks shape: {}".format(landmarks.shape))
print("First 4 landmarks: {}".format(landmarks[:4]))

def show_landmarks(image, landmarks):
    """show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:,0], landmarks[:,1], s=10, marker='.', c='r')
    plt.pause(0.001)

plt.figure()
show_landmarks(io.imread("/Users/pengyuyan/Desktop/pytorch/data/faces/"+img_name), landmarks)
plt.show()

# 下面介绍一下pytorch内置的Dataset类，torch.utils.data.Dataset，
# 所以之后自己定制的数据集都会继承这个类所具有的特性，比如
# __len__ : len(dataset)会返回数据集的大小
# __getitem__ : dataset[i]可以返回数据集中的第i个样本
# 接下来就为人脸新建立一个数据集，他会在__init__中读入csv，
# 保留__getitem__来读取图片数据，数据集中的样本将会是一个字典的样子，像
# {'image': image, 'landmarks': landmarks}，同时加入transform方法，
# 使得能够对数据进行预处理

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset.""" 
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.landmarks_frame)
    
    def __getitem__(self, idx):
        img_name = "/Users/pengyuyan/Desktop/pytorch/data/faces/"+self.landmarks_frame.iloc[idx, 0]
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype(float).reshape(-1, 2)
        sample = {"image":image, "landmarks":landmarks}

        if self.transform:
            sample = self.transform(sample)
        
        return sample


face_dataset = FaceLandmarksDataset("/Users/pengyuyan/Desktop/pytorch/data/faces/face_landmarks.csv",
                                    "/Users/pengyuyan/Desktop/pytorch/data/faces/")

fig = plt.figure()

for i in range(len(face_dataset)):
    sample = face_dataset[i]
    print(i, sample["image"].shape, sample["landmarks"].shape)
    ax = plt.subplot(1, 4, i+1)
    ax.set_title("Sample #{}".format(i))
    ax.axis("off")
    # **kwargs 允许你将不定长度的键值对, 作为参数传递给一个函数。 
    # 如果你想要在一个函数里处理带名字的参数, 你应该使用**kwargs。
    show_landmarks(**sample)
    if i==3:
        plt.show()
        break

# 可以发现上面的那些图片大小是不一样的，但是一些网络要求输入图片的大小是一致的，
# 所以我们需要对图片进行一些预处理，比如scale，crop等等
# 我们选择将他们写成callable classes而非简单的函数，
# 这样我们就不需要每次用这个函数的时候就需要给它传一次参数了
# 然后就可以像下面这样使用transform了
tsfm = Transform(param)
transformed_sample = tsfm(sample)

class Rescale(object):
    """Rescale the image into a given size

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self,output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    
    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size* h / w , self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size* w / h
        else:
            new_h, new_w = self.output_size
        
        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks*[new_w / w, new_h / h]

        return {"image":image, "landmarks":landmarks}
    
class Crop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int,tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        h, w = image.shape[:2]

        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top : top + new_h, left : left + new_w, :]
        landmarks = landmarks - [left, top]

        return {"image":image, "landmarks":landmarks}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2,0,1))

        return {"image":torch.from_numpy(image),"landmarks":torch.from_numpy(landmarks)}

# Compose transforms
scale = Rescale(256)
crop = Crop(128)
composed = transforms.Compose([Rescale(256),
                              Crop(128)])
# Apply each of the above transforms on sample.
fig = plt.figure()
sample = face_dataset[65]
for i, tsfm in enumerate([scale, crop, composed]):
    transformed_sample = tsfm(sample)

    ax = plt.subplot(1, 3, i+1)
    plt.tight_layout()
    ax.set_title(type(tsfm).__name__)
    show_landmarks(**transformed_sample)

plt.show()

transformed_dataset = FaceLandmarksDataset(
    csv_file="/Users/pengyuyan/Desktop/pytorch/data/faces/face_landmarks.csv",
    root_dir="/Users/pengyuyan/Desktop/pytorch/data/faces/",
    transform=transforms.Compose([
        Rescale(256),
        Crop(128),
        ToTensor()
    ]))

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]
    print(i, sample["image"].size(), sample["landmarks"].size())

    if i==3:
        break


# dataloader = Dataloader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)
dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=4)

def show_landmarks_batch(sample_batched):
    """show image with landmarks for a batch of samples"""
    images_batch, landmarks_batch = sample_batched["image"], sample_batched["landmarks"]
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i*im_size,
                    landmarks_batch[i, :, 1].numpy())
        plt.title('Batch from dataloader')

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched["image"].size(),
          sample_batched["landmarks"].size())

    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break

    





