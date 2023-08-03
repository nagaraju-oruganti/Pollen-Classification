### libraries
import os
import pandas as pd
import numpy as np
import random
import glob

## PyTorch and Image
import torch
from torchvision import transforms
from PIL import Image as im
from torch.utils.data import Dataset, DataLoader
import tifffile as tiff

## sklearn
from sklearn.model_selection import train_test_split

## ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Load data and split into test train
def split_dataset(data_dir, seed, save = True):
    paths = glob.glob(f'{data_dir}/*/*.tif')
    data = [{'file': os.path.basename(p), 'class': os.path.basename(os.path.dirname(p))} for p in paths]
    print(len(data))
    
    # Make dataframe
    df = pd.DataFrame(data)
    print(df.head())
    
    # split into train and test
    train_df, test_df = train_test_split(df, stratify = df['class'], test_size=0.2,random_state=42)
    train_df['kind'] = 'train'
    test_df['kind'] = 'test'
    df = pd.concat([train_df, test_df])
    df.reset_index(inplace = True, drop = True)

    if save:
        df.to_csv(f'{data_dir}/fold_map.csv')
    
    return df

# Preprocess dataset
class Preprocess:
    def __init__(self, config):
        self.config = config
        self.data_dir = config.data_dir
        self.seed = config.seed

    # split dataset into train and test
    def split_data(self):
        
        folds_path = os.path.join(self.data_dir, 'fold_map.csv')
        
        # Load fold if exists
        if os.path.exists(folds_path):
            self.df = pd.read_csv(folds_path)
        else:
            self.df = split_dataset(data_dir=self.data_dir, seed = self.seed)
            
        # Map class labels
        self.INDEX_TO_LABEL_MAP = {idx:label for idx, label in enumerate(self.df['class'].unique())}
        self.LABEL_TO_INDEX_MAP = {label:idx for idx, label in self.INDEX_TO_LABEL_MAP.items()}
        
    # Create new sample from train images for training
    def augmentation(self):
        ''
    
    def make_datasets(self):
        
        # split data
        self.split_data()
        
        # make datasets
        data = {'train': [], 'test': []}
        for kind in data:
            df = self.df[self.df['kind'] == kind]
            df['label'] = df['class'].map(self.LABEL_TO_INDEX_MAP)
            
            for _, row in df.iterrows():
                class_name  = row['class']
                filename    = row['file']
                label       = row['label']
                
                # image_path
                image_path = f'{class_name}/{filename}'
                
                # append to the dataset
                data[kind].append((image_path, label, False))       # image path, label, and is_augment (default no augmentation)
                
        self.data = data



class PollenDataset(Dataset):
    def __init__(self, config, data):
        self.data = data
        self.data_dir = config.data_dir
        self.height = config.height
        self.width = config.width
        self.max_rotation_angle = config.max_rotation_angle
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.read_image(self.data[idx][0])
        label = self.data[idx][1]
            
        img_tensor = self.apply_transformation(img = image, augment = self.data[idx][2])
        label_tensor = torch.tensor(label, dtype= torch.long)
        
        return img_tensor, label_tensor
        
    def read_image(self, img_path):
        return tiff.imread(f'{self.data_dir}/{img_path}')
    
    def apply_transformation(self, img, augment):
        num_frames = img.shape[0]
        transformed_image = torch.empty(num_frames, 3, self.height, self.width)
        transform = self.make_transform(augment)
        for i in range(num_frames):
            transformed_image[i, :, :, :] = transform(img[i, :, :])   # apply transformation and save
        return transformed_image
    
    def make_transform(self, augment = False):
        transform = transforms.Compose([
            transforms.ToPILImage(),                            # convert tiff frame to PIL Image
            transforms.Grayscale(num_output_channels=3),        # convert greyscale image to RGB
            transforms.Resize((self.height, self.width)),       # Resize frames to a specific size
            transforms.ToTensor(),                              # Convert frames to tensors
            
            # images are in grey scale and already normalized between 0 and 1
            # otherwise uncomment to normalize the pixels
            # transforms.Lambda(lambda x: x/255.0)    # Normalize pixel between 0, 1
            
            ## Augmentation
            transforms.RandomHorizontalFlip(p = 0.8 if augment else 0),             # Randomly apply horizontal flip with probability 0.8
            transforms.RandomRotation(self.max_rotation_angle if augment else 0), 
            
        ])
        return transform
        
    
# Dataloader
def dataloaders(cfg):

    prep = Preprocess(config=cfg)
    prep.make_datasets()
    
    train_dataset = prep.data['train']
    
    # Augmentation
    aug_dataset = []
    for d in train_dataset:
        if random.random() >= 1 - cfg.aug_threshold:        # aug_threshold of > 1 means no augmentation
            aug_dataset.append((d[0], d[1], True))
    train_dataset += aug_dataset
    
    train = PollenDataset(cfg, prep.data['train'])
    aug_train = PollenDataset(cfg, train_dataset)
    valid = PollenDataset(cfg, prep.data['test'])

    train_loader = DataLoader(train,
                              batch_size=cfg.train_batch_size,
                              shuffle=True,
                              drop_last=False)
    
    aug_train_loader = DataLoader(aug_train,
                                  batch_size=cfg.train_batch_size,
                                  shuffle=True,
                                  drop_last=False)

    valid_loader = DataLoader(valid,
                              batch_size=cfg.valid_batch_size,
                              shuffle=False)

    return train_loader, aug_train_loader, valid_loader