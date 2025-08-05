import os
import torch
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import exists
from PIL import Image as im
from torchvision import transforms
from torch.utils.data import Dataset
from pathlib import Path
from functools import lru_cache

def img_train_transform():
    train_transform_list = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return train_transform_list

def img_val_transform():
    val_transform_list = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    return val_transform_list  

def img_val_transform_random_ten_crop():
    val_transform_list = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    return val_transform_list  

# def img_val_transform_ten_crop():
#     val_transform_list = transforms.Compose([
#             transforms.Resize(256),
#             transforms.PILToTensor(),
#             transforms.ConvertImageDtype(torch.float),
#             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#             transforms.TenCrop(224),
#             transforms.Lambda(lambda crops: torch.stack([crop for crop in crops]))
#         ])
#     return val_transform_list  

def img_val_transform_ten_crop():
    m16_transform_list = transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(crop) for crop in crops]))
        ])
    return m16_transform_list

def simclr_transform():
    simclr_transform_list = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return simclr_transform_list


class GeoDataLoader(Dataset):
    """
    DataLoader for image-gps datasets.
    
    The expected CSV file with the dataset information should have columns:
    - 'IMG_FILE' for the image filename,
    - 'LAT' for latitude, and
    - 'LON' for longitude.
    
    Attributes:
        dataset_file (str): CSV file path containing image names and GPS coordinates.
        dataset_folder (str): Base folder where images are stored.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, dataset_file, dataset_folder, transform=None):
        self.dataset_folder = dataset_folder
        self.transform = transform
        self.images, self.coordinates = self.load_dataset(dataset_file)

    def load_dataset(self, dataset_file):
        try:
            dataset_info = pd.read_csv(dataset_file)
        except Exception as e:
            raise IOError(f"Error reading {dataset_file}: {e}")

        images = []
        coordinates = []

        for _, row in tqdm(dataset_info.iterrows(), desc="Loading image paths and coordinates"):
            try:
                filename = row['IMG_FILE']
                if exists(filename):
                    images.append(filename)
                    latitude = float(row['LAT'])
                    longitude = float(row['LON'])
                    coordinates.append((latitude, longitude))
            except:
                filename = row['img_file']
                if exists(filename):
                    images.append(filename)
                    latitude = float(row['lat'])
                    longitude = float(row['lon'])
                    coordinates.append((latitude, longitude))
        # coordinates = torch.cat(coordinates, dim=0)
        coordinates = torch.tensor(coordinates)

        return images, coordinates

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        gps = self.coordinates[idx]

        image = im.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        return image, gps
    

class GeoDataLoaderRandomTenCrop(GeoDataLoader):
    """
    DataLoader for image-gps datasets with TenCrop augmentation.
    """
    def __getitem__(self, idx):
        img_path = self.images[idx]
        gps = self.coordinates[idx]

        # Load image
        image = im.open(img_path).convert('RGB')
        
        # Apply the transform to get 5 crops
        crops = [self.transform(image) for _ in range(5)]
        
        # Create 5 more augmentations by flipping the original 5 crops
        flipped_crops = [torch.flip(crop, [2]) for crop in crops]  # Assuming the 2nd dimension is width
        
        # Combine original and flipped crops
        all_crops = crops + flipped_crops
        
        # Stack all crops to get a tensor of the desired shape: (10, 3, H, W)
        # Note: This assumes the transform returns a tensor of shape (3, H, W)
        stacked_crops = torch.stack(all_crops)

        # stacked_crops = self.transform(image)
        
        return stacked_crops, gps
    

class GeoDataLoaderTenCrop(GeoDataLoader):
    """
    DataLoader for image-gps datasets with TenCrop augmentation.
    """
    
    # @lru_cache(maxsize=2880)
    def __getitem__(self, idx):
        img_path = self.images[idx]
        gps = self.coordinates[idx]

        # Load image
        image = im.open(img_path).convert('RGB')
        
        # Apply the transform to get 5 crops
        # crops = [self.transform(image) for _ in range(5)]
        
        # Create 5 more augmentations by flipping the original 5 crops
        # flipped_crops = [torch.flip(crop, [2]) for crop in crops]  # Assuming the 2nd dimension is width
        
        # Combine original and flipped crops
        # all_crops = crops + flipped_crops
        
        # Stack all crops to get a tensor of the desired shape: (10, 3, H, W)
        # Note: This assumes the transform returns a tensor of shape (3, H, W)
        # stacked_crops = torch.stack(all_crops)

        stacked_crops = self.transform(image)
        
        return stacked_crops, gps


import pdb
class GeoDataLoaderPrecomputed(Dataset):
    def __init__(self, dataset_dir, dataset_sample=1.0, seed=23):
        self.root = dataset_dir
        self.dataset_sample = dataset_sample
        self.seed = seed

        self.rng = np.random.default_rng(seed=self.seed)

        self.n_augs = sum(os.path.isdir(os.path.join(self.root, name)) 
                          for name in os.listdir(self.root))
        
        files = os.listdir(os.path.join(dataset_dir, '0'))
        self.tensors = sorted([int(file[1:-3]) for file in files if file.startswith('G')])
            
    def __len__(self):
        return len(self.tensors)
    
    def __getitem__(self, index):
        # Choose one augmentation randomly
        augmentation = str(self.rng.integers(0, self.n_augs))

        # Get tensors
        features_num = self.tensors[index]
        features_file = 'I' + str(features_num) + '.pt'
        labels_file = 'G' + str(features_num) + '.pt'
        features_path = os.path.join(self.root, augmentation, features_file)
        labels_path = os.path.join(self.root, augmentation, labels_file)

        features = torch.load(features_path, map_location=torch.device('cpu'))
        labels = torch.load(labels_path, map_location=torch.device('cpu'))

        # print(f"mp16_batch_features.shape = {features.shape}", flush=True)

        return features, labels
        
        
class MapillaryDataset(Dataset):
    def __init__(self, dataset_dir, n_views=1, n_batches=4, dataset_sample=1.0, seed=23):
        super().__init__()

        self.root = dataset_dir
        self.n_views = n_views
        self.n_batches = n_batches
        self.samples_per_batch = 128
        self.dataset_sample = dataset_sample
        self.seed = seed

        self.features_dir = os.path.join(self.root, 'features')
        self.labels_dir = os.path.join(self.root, 'labels')

        self.rng = np.random.default_rng(seed=self.seed)

        # Find number of augmentations
        self.n_augs = sum(os.path.isdir(os.path.join(self.features_dir, name)) 
                        for name in os.listdir(self.features_dir))
            
        if self.n_views > self.n_augs:
            raise ValueError(f"n_views ({self.n_views}) must be lower than n_augs ({self.n_augs})")
        
        # Create list with the relative path of all features/labels
        features_directory = Path(os.path.join(self.features_dir, '0'))
        feature_paths = list(features_directory.rglob('*.npy'))
        # self.tensors = sorted([file_path.relative_to(features_directory) for file_path in feature_paths])

        all_tensors = sorted([file_path.relative_to(features_directory) for file_path in feature_paths])

        if dataset_sample < 1:
            num_samples = int(len(all_tensors) * dataset_sample)
            sampled_indices = self.rng.choice(len(all_tensors), size=num_samples, replace=False)
            self.tensors = [all_tensors[i] for i in sampled_indices]
        else:
            self.tensors = all_tensors
        
    def __len__(self):
        return len(self.tensors) // self.n_batches
    
    def __getitem__(self, index):
        # Choose two augmentations randomly
        augs = self.rng.choice(self.n_augs, size=self.n_views, replace=False).tolist()

        batch_features = []
        batch_labels = []

        for i in range(self.n_batches):
            idx = index * self.n_batches + i
            features = []
            for aug in augs:
                features_path = os.path.join(self.features_dir, str(aug), self.tensors[idx])
                #import pdb;pdb.set_trace()
                features_np = np.load(features_path)
                features_tensor = torch.from_numpy(features_np)
                #print(f"{features_tensor.shape=}")
                features.append(features_tensor)
                   
            features = torch.stack(features, dim=1)
            #print(f"{features.shape=}")
            features = features.view(-1, features.size(2))  # (256, 768)
            #print(f"{features.shape=}")

            labels_path = os.path.join(self.labels_dir, '0', self.tensors[idx])
            labels_np = np.load(labels_path)
            labels = torch.from_numpy(labels_np)            # (128, 7)
            #print(f"{labels.shape=}")

            batch_features.append(features)
            batch_labels.append(labels)

        batch_features = torch.cat(batch_features, dim=0)   # (1024, 768)
        batch_labels = torch.cat(batch_labels, dim=0)       # (512, 7)
        #print(f"{batch_features.shape=}")
        #print(f"{batch_labels.shape=}")

        #time = batch_labels[:, :5]
        gps = batch_labels
       

        #dataset = 'mapillary'
                
        return batch_features, gps#, time, dataset
    

class MapillaryTenCropDataset(Dataset):
    def __init__(self, dataset_dir, n_batches=4, dataset_sample=1.0, seed=23):
        super().__init__()

        self.root = dataset_dir
        self.n_batches = n_batches
        self.samples_per_batch = 128
        self.dataset_sample = dataset_sample
        self.seed = seed

        self.features_dir = os.path.join(self.root, 'features')
        self.labels_dir = os.path.join(self.root, 'labels')
        
        # Create list with the relative path of all features/labels
        #features_directory = 
        feature_paths = list(Path(os.path.join(self.features_dir, '0')).rglob('*.npy')) + list(Path(os.path.join(self.features_dir, '1')).rglob('*.npy'))
        # self.tensors = sorted([file_path.relative_to(features_directory) for file_path in feature_paths])

        # all_tensors = sorted([file_path.relative_to(features_directory) for file_path in feature_paths])
        all_tensors = ["/".join(str(x).split("/")[-2:]) for x in feature_paths]
        # print(all_tensors[:5])
        # print(all_tensors[-5:])
        self.tensors = all_tensors
        
    def __len__(self):
        return len(self.tensors) // self.n_batches
    
    def __getitem__(self, index):

        batch_features = []
        batch_labels = []

        for i in range(self.n_batches):
            idx = index * self.n_batches + i
            features_path = os.path.join(self.features_dir, self.tensors[idx])
            features_np = np.load(features_path)
            features = torch.from_numpy(features_np)
                   
            # features = torch.stack(features, dim=1)
            # #print(f"{features.shape=}")
            features = features.view(-1, features.size(2))  # (256, 768)
            #print(f"{features.shape=}")

            labels_path = os.path.join(self.labels_dir, self.tensors[idx])
            labels_np = np.load(labels_path)
            labels = torch.from_numpy(labels_np) 

            batch_features.append(features)
            batch_labels.append(labels)

        batch_features = torch.cat(batch_features, dim=0)   # (1280, 768)
        batch_labels = torch.cat(batch_labels, dim=0)       # (128, 2)

        # print(batch_features.shape)
        # print(batch_labels.shape)
                
        return batch_features, batch_labels

class MapillaryTemporalDataset(Dataset):
    def __init__(self, dataset_dir, pred_labels_dir, train_or_val='train', n_frames=4, gap=3, dataset_sample=1.0, seed=23):
        super().__init__()

        self.root = dataset_dir
        self.n_frames = n_frames
        self.dataset_sample = dataset_sample
        self.seed = seed
        self.train_or_val = train_or_val
        self.gap = gap

        self.features_dir = os.path.join(self.root, 'features')
        self.labels_dir = os.path.join(self.root, 'labels')
        self.pred_labels_dir = pred_labels_dir

        if train_or_val == 'train':
            self.rng = np.random.default_rng(seed=self.seed)

            self.n_augs = sum(os.path.isdir(os.path.join(self.features_dir, name)) 
                            for name in os.listdir(self.features_dir))
        
            # Create list with the relative path of all features/labels
            features_directory = Path(os.path.join(self.features_dir, '0'))

        else:
            features_directory = Path(self.features_dir)

        feature_paths = list(features_directory.rglob('*.npy'))
        # print(f"{feature_paths=}")
        all_tensors = sorted([file_path.relative_to(features_directory) for file_path in feature_paths])

        if dataset_sample < 1:
            num_samples = int(len(all_tensors) * dataset_sample)
            sampled_indices = self.rng.choice(len(all_tensors), size=num_samples, replace=False)
            self.tensors = [all_tensors[i] for i in sampled_indices]
        else:
            self.tensors = all_tensors

        self.seq_ids = [str(x).split(".")[0] for x in self.tensors]
        
    def __len__(self):
        return len(self.tensors) 
    
    def __getitem__(self, index):
        # Choose two augmentations randomly
        if self.train_or_val == 'train':
            aug = self.rng.choice(self.n_augs, size=1, replace=False).tolist()[0]
            features_path = os.path.join(self.features_dir, str(aug), self.tensors[index])
            labels_path = os.path.join(self.labels_dir, '0', self.tensors[index])
        else:
            features_path = os.path.join(self.features_dir, self.tensors[index])
            labels_path = os.path.join(self.labels_dir, self.tensors[index])

        pred_labels_path = os.path.join(self.pred_labels_dir, self.tensors[index])

        features = torch.from_numpy(np.load(features_path))
        labels = torch.from_numpy(np.load(labels_path))
        
        if self.train_or_val == 'val':
            pred_labels = torch.from_numpy(np.load(pred_labels_path))[:,0,:]

        if self.train_or_val == 'train':
            frame_count = features.shape[0]        
            gap = self.gap #rng.choice(3, size=1, replace=False).tolist()[0] + 1
            remaining = frame_count - gap*self.n_frames
            if remaining > 0:
                start = self.rng.choice(remaining, size=1, replace=False).tolist()[0]
            else:
                gap = gap // 2
                remaining = frame_count - gap*self.n_frames
                if not remaining <= 0:
                    start = self.rng.choice(remaining, size=1, replace=False).tolist()[0]
                else:
                    start = 0

            indices = list(range(start,start + gap*self.n_frames,gap))

            # print(indices)

            # print(index, indices)
            # print(self.seq_ids[index])
            # print(features[indices])
            # print(labels)
            # print(len(labels))
            # print(len(labels[0]))
            # print(len(labels[0][0]))
            # if isinstance(labels[0][0], list):
            #     assert len(labels) == 1
            #     labels = labels[0]
            if len(labels.shape) == 3:
                assert labels.shape[0] == 1
                labels = labels[0]  

            pred_labels_final = labels[indices]

            #Jitter
            min_noise = 0.001#0.05
            max_noise = 0.02#0.2

            lat_noise = torch.FloatTensor(len(indices), 1).uniform_(min_noise, max_noise)
            lat_noise *= torch.Tensor(np.random.choice([1,-1], (len(indices), 1)))
            lon_noise = torch.FloatTensor(len(indices), 1).uniform_(min_noise, max_noise)
            lon_noise *= torch.Tensor(np.random.choice([1,-1], (len(indices), 1)))

            #Shift
            max_noise = 0.2

            lat_noise += torch.ones(len(indices),1)*torch.FloatTensor(1).uniform_(-1*max_noise, max_noise)
            lon_noise += torch.ones(len(indices),1)*torch.FloatTensor(1).uniform_(-1*max_noise, max_noise)
            full_noise = torch.cat((lat_noise, lon_noise), dim=1)
            pred_labels_final += full_noise
            
            #Collapse
            if torch.FloatTensor(1).uniform_(0, 1).item() < 0.1:
                pred_labels_final = torch.vstack([pred_labels_final[np.random.choice([i for i in range(len(indices))], 1).item()]]*len(indices))           
    
            return self.seq_ids[index], features[indices], pred_labels_final, labels[indices]
        else:
            return self.seq_ids[index], features, pred_labels, labels




def create_mapillary_precomp_datasets(dataset_dir, n_views=2, n_batches=4, dataset_sample=1.0, seed=23):
    return {
        'train': MapillaryDataset(os.path.join(dataset_dir, 'train'), n_views, n_batches, dataset_sample, seed),
        'val': MapillaryDataset(os.path.join(dataset_dir, 'val'), 1, n_batches, dataset_sample, seed),
        'test': None,
    }

def temporal_flat_collate_fn(batch_data):


    feats = []
    labels = []

    for (seq_id, feat, label) in batch_data:
        feats.append(feat)
        labels.append(label)
        C = feat.shape[-1]

    return torch.stack(feats).reshape(-1,C), torch.stack(labels).mean(dim=1)

def temporal_collate_fn(batch_data):


    feats = []
    pred_labels = []
    labels = []

    for (seq_id, feat, pred_label, label) in batch_data:
        feats.append(feat)
        pred_labels.append(pred_label)
        labels.append(label)
        C = feat.shape[-1]

    return torch.stack(feats), torch.stack(pred_labels), torch.stack(labels)


def temporal_single_collate_fn(batch_data):
    seq_id = batch_data[0][0]
    feats = batch_data[0][1]
    pred_labels = batch_data[0][2]
    labels = batch_data[0][3][0]

    Tf, V, C = feats.shape
    Tg, D = labels.shape
    feats = feats.view(Tf*V, C)

    feats = feats.unsqueeze(0)
    pred_labels = pred_labels.unsqueeze(0)
    labels = labels.unsqueeze(0)

    assert D == 2
    assert Tf == Tg
    assert V == 10

    return seq_id, feats, pred_labels, labels




if __name__ == '__main__':
    # ds = MapillaryDataset(dataset_dir = '/home/c3-0/pr288313/datasets/Mapillary/tensors/train', n_views=2)

    from tqdm import tqdm
    from torch.utils.data import DataLoader


    
    

    ds = MapillaryTemporalDataset('/home/c3-0/pr288313/datasets/Mapillary/tensors_video_seq/train', '../predictions_dino/2000/2000', train_or_val='train', n_frames=2, dataset_sample=1.0, seed=23)
    # ds = MapillaryTemporalDataset('../../feat_extarct/features/Mapillary/tensors_video_seq/val', train_or_val='val', n_frames=None, dataset_sample=1.0, seed=23)

    B = 1


    dl = DataLoader(
        dataset=ds,
        batch_size=B,
        shuffle=False,
        num_workers=0,
        collate_fn = temporal_collate_fn
    )
    print(f'{len(ds)=}')

    segments = []

    for frames, pred_labels, labels in dl:
        # B, T, C = frames.shape
        # frames_flat = frames.reshape(-1,C)
        # labels_flat = labels.mean(dim=1)
        print(frames.shape, pred_labels.shape, labels.shape)

    # print(ds[0])

    # for i in tqdm(range(len(ds))):
    #     feat, label = ds[i]
    #     if feat.shape[0] != 4 or label.shape[0] !=4:
    #         print(i, feat.shape, label.shape)
