
from torchvision.transforms import transforms
import PIL.Image as Image
from torch.utils.data import ConcatDataset
import torch
from sklearn.utils import class_weight
from torch.utils.data import (
    Dataset,
    DataLoader,
)
import config
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.io import imread
from tqdm import tqdm

IMAGE_SIZE = 28


from skimage.transform import rotate
from skimage.util import random_noise



def do_augmentation(data, path, ext='.jpg', file_name='augmentation.csv', dir='augmentation'):
    data = pd.read_csv(data)
    max = data['level'].value_counts().max()

    augmentation = {}
    augmentation_cnt = {}
    for index, value in data['level'].value_counts().items():
        augmentation[index] = max - value
        augmentation_cnt[index] = 0
    print(augmentation)
    print(augmentation_cnt)


    imgs = []
    for img_name in tqdm(data['id_code']):
        image_path =  path+ img_name + ext
        img = imread(image_path)
        img = img / 255
        imgs.append(img)

    x = np.array(imgs)
    y = data['level'].values

    augmentation_dict = {}
    for i in tqdm(range(x.shape[0])):
        if augmentation[y[i]] > augmentation_cnt[y[i]]:
            plt.imsave(dir+'/1_' + str(i) + '_augmentation'+ext, rotate(x[i], angle=45, mode='wrap'))
            plt.imsave(dir+'/2_' + str(i) + '_augmentation'+ext, rotate(x[i], angle=45, mode='wrap'))
            plt.imsave(dir+'/3_' + str(i) + '_augmentation'+ext, np.fliplr(x[i]))
            plt.imsave(dir+'/4_' + str(i) + '_augmentation'+ext, np.flipud(x[i]))
            plt.imsave(dir+'/5_' + str(i) + '_augmentation'+ext, random_noise(x[i], var=0.2 ** 2))

            augmentation_dict['1_' + str(i) + '_augmentation'] = y[i]
            augmentation_dict['2_' + str(i) + '_augmentation'] = y[i]
            augmentation_dict['3_' + str(i) + '_augmentation'] = y[i]
            augmentation_dict['4_' + str(i) + '_augmentation'] = y[i]
            augmentation_dict['5_' + str(i) + '_augmentation'] = y[i]

            augmentation_cnt[y[i]] = augmentation_cnt[y[i]] + 5

    data_items = augmentation_dict.items()
    data_list = list(data_items)
    df = pd.DataFrame(data_list, columns=['id_code', 'level'])
    df.to_csv(file_name, index=False)



class dataset(Dataset):

    def __init__(self, df, data_path, image_transform=None, train=True):  # Constructor.
        super(Dataset, self).__init__()  # Calls the constructor of the Dataset class.
        self.df = df
        self.data_path = data_path
        self.image_transform = image_transform
        self.train = train

    def __len__(self):
        return len(self.df)  # Returns the number of samples in the dataset.

    def __getitem__(self, index):
        image_id = self.df['id_code'][index]

        try:
            image = Image.open(f'{self.data_path}/{image_id}.jpg')  # Image.
        except FileNotFoundError:
            image = Image.open(f'{self.data_path}/{image_id}.jpeg')  # Image.

        if self.image_transform:
            image = self.image_transform(image)  # Applies transformation to the image.

        if self.train:
            label = self.df['level'][index]  # Label.
            return image, label  # If train == True, return image & label.

        else:
            return image  # If train != True, return image.


def get_weight(data_lebel, n_classes=2, graph=False, data_aug=None):
    if n_classes == -3:
        if not (data_aug is None):
            data_lebel = pd.concat([data_lebel, data_aug])

        data_lebel.drop(data_lebel.loc[data_lebel['level'] == 0].index, inplace=True)
        data_lebel.drop(data_lebel.loc[data_lebel['level'] == 1].index, inplace=True)

        data_lebel["level"].replace({2: 0, 3: 1, 4: 2 }, inplace=True)
        data_lebel = data_lebel.reset_index()
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.array([0, 1, 2]),
                                                          y=data_lebel['level'].values)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(config.DEVICE)

    if n_classes == 0:

        data_lebel = data_lebel.reset_index()
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.array([0, 1, 2, 3, 4]),
                                                          y=data_lebel['level'].values)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(config.DEVICE)

    if n_classes == 2:
        data_lebel["level"].replace({2: 1, 3: 1, 4: 1}, inplace=True)
        data_lebel = data_lebel.reset_index()
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.array([0, 1]),
                                                          y=data_lebel['level'].values)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(config.DEVICE)
    if n_classes == 3:
        data_lebel["level"].replace({3: 2, 4: 2}, inplace=True)
        data_lebel = data_lebel.reset_index()
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.array([0, 1, 2]),
                                                          y=data_lebel['level'].values)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(config.DEVICE)

    if n_classes == 4:

        if not (data_aug is None):
            data_lebel = pd.concat([data_lebel, data_aug])

        data_lebel.drop(data_lebel.loc[data_lebel['level'] == 0].index, inplace=True)
        data_lebel["level"].replace({1: 0, 2: 1, 3: 2, 4: 3}, inplace=True)
        data_lebel = data_lebel.reset_index()
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.array([0, 1, 2, 3]), y=data_lebel['level'].values)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(config.DEVICE)


    if graph:


        ax = data_lebel['level'].value_counts().plot(kind='bar',
                                            figsize=(14, 8),
                                            title="Class")
        ax.set_xlabel("Levels")
        ax.set_ylabel("Frequency")
        plt.show()

    if not (data_aug is None):
        df_row = pd.concat([data_lebel, data_aug])

        ax = df_row['level'].value_counts().plot(kind='bar',
                                                     figsize=(14, 8),
                                                     title="Class")
        ax.set_xlabel("Levels")
        ax.set_ylabel("Frequency")
        plt.show()

    return class_weights


def get_dataset(path, data_lebel):
    test_transform = transforms.Compose([transforms.Resize([720, 720]),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                         ])

    data_set = dataset(data_lebel, f'{path}train', image_transform=test_transform)

    train_size = int(0.8 * len(data_set))
    val_size = len(data_set) - train_size

    train_set, valid_set = torch.utils.data.random_split(data_set, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train = DataLoader(train_set, batch_size=config.BATCH, shuffle=True)
    valid = DataLoader(valid_set, batch_size=config.BATCH, shuffle=False)
    return train, valid






def get_data_loader_2_classes(path, data_lebel, size):

    test_transform = transforms.Compose([transforms.Resize([size, size]),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                         ])

    data_lebel["level"].replace({2: 1, 3: 1, 4: 1}, inplace=True)
    train_df = data_lebel.reset_index()
    data_set = dataset(train_df, f'{path}train', image_transform=test_transform)

    train_size = int(0.8 * len(data_set))
    val_size = len(data_set) - train_size

    train_set, valid_set = torch.utils.data.random_split(data_set, [train_size, val_size],
                                                         generator=torch.Generator().manual_seed(42))
    train = DataLoader(train_set, batch_size=config.BATCH, shuffle=True)
    valid = DataLoader(valid_set, batch_size=config.BATCH, shuffle=False)

    return train, valid




def get_data_loader_4_classes(path, data_lebel, size, aug = None,  aug_path=None):

    test_transform = transforms.Compose([transforms.Resize([size, size]),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                         ])

    data_lebel.drop(data_lebel.loc[data_lebel['level'] == 0].index, inplace=True)
    data_lebel["level"].replace({1: 0, 2: 1, 3: 2, 4: 3}, inplace=True)


    aug.drop(aug.loc[aug['level'] == 0].index, inplace=True)
    aug["level"].replace({1: 0, 2: 1, 3: 2, 4: 3}, inplace=True)

    data_lebel.reset_index(inplace=True)
    aug.reset_index(inplace=True)

    #train_df = pd.concat([data_lebel, aug])


    data_set = dataset(data_lebel, f'{path}train', image_transform=test_transform)

    data_set_aug = dataset(aug, f'{aug_path}', image_transform=test_transform)


    data_set = ConcatDataset([data_set_aug, data_set])




    train_size = int(0.9 * len(data_set))
    val_size = len(data_set) - train_size

    train_set, valid_set = torch.utils.data.random_split(data_set, [train_size, val_size],
                                                         generator=torch.Generator().manual_seed(42))
    train = DataLoader(train_set, batch_size=config.BATCH, shuffle=True)
    valid = DataLoader(valid_set, batch_size=config.BATCH, shuffle=False)

    return train, valid


def get_test_dataset_4_classes(path, data_lebel, size):
    test_transform = transforms.Compose([transforms.Resize([size, size]),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                         ])
    data_lebel.drop(data_lebel.loc[data_lebel['level'] == 0].index, inplace=True)
    data_lebel["level"].replace({1: 0, 2: 1, 3: 2, 4: 3}, inplace=True)

    data_lebel = data_lebel.reset_index()
    data_set = dataset(data_lebel, f'{path}test', image_transform=test_transform)
    test = DataLoader(data_set, batch_size=config.BATCH, shuffle=False)
    return test




def get_test_dataset_2_classes(path, data_lebel, size):
    test_transform = transforms.Compose([transforms.Resize([size, size]),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                         ])
    data_lebel["level"].replace({2: 1, 3: 1, 4: 1}, inplace=True)
    data_lebel = data_lebel.reset_index()
    data_set = dataset(data_lebel, f'{path}test', image_transform=test_transform)
    test = DataLoader(data_set, batch_size=config.BATCH, shuffle=False)
    return test


def get_test_dataset(path, data_lebel, size):
    test_transform = transforms.Compose([transforms.Resize([size, size]),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                         ])
    data_set = dataset(data_lebel, f'{path}test', image_transform=test_transform)
    test = DataLoader(data_set, batch_size=config.BATCH, shuffle=False)
    return test