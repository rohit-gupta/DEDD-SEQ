import os
import contextlib
import math
from multiprocessing import Pool
import gc
from pathlib import Path

import pandas as pd
import torch
from torch import autocast
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.transforms as T

from models.image_encoder import CLIPVisualEncoder, SiglipMLVisualEncoder
from data.transforms import get_clip_transforms

from dataset import GeoSeqDataLoader

CONFIG = {
    "output_dir": "/home/parthpk/XGeoCLIP/geo-clip_new/feat_extarct/features/CityGuessr_feats/tensors_video_seq_DINOCLIP/train/",
    "vis_bb": "DINOCLIP",
    "batch_size": 1,
    "dataset_sample": 1.0,
    "seed": 23,
    "num_workers": 0,
}


def main():

    split = 'train'
    #split = 'train_tencrop'



    ##############################################
    # SELECT DEVICE
    ##############################################

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device:         {device}")
    print("")

    
    gc.collect()
    torch.cuda.empty_cache()

    ##############################################
    # CREATE MODEL
    ##############################################

    if CONFIG['vis_bb'] == 'CLIP':
        model = CLIPVisualEncoder(architecture="ViT-L/14", freeze_weights=True)
    elif CONFIG['vis_bb'] == 'DINO': 
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
    elif CONFIG['vis_bb'] == 'SiglipML':
        model = SiglipMLVisualEncoder(freeze_weights=True)
    elif CONFIG['vis_bb'] == 'DINOCLIP':
        models = [CLIPVisualEncoder(architecture="ViT-L/14", freeze_weights=True), torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')]
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
    

    models = [model.to(device) for model in models]
    for model in models:
        model.eval()

    ##############################################
    # DATALOADER
    ##############################################

    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)

    transform_dict = {}
    transform_dict['val'] = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.TenCrop(224),
            transforms.Lambda(
                lambda crops: torch.stack(
                    [transforms.ToTensor()(crop) for crop in crops]
                )
            ),
            transforms.Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize(mean, std)(crop) for crop in crops]
                )
            ),
        ]
    )

    transform_dict['train'] = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.6, 1.0), ratio=(0.999, 1.001), antialias=True),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean, std),
    ])

    
    if split == 'train_tencrop':
        msls_dataset = GeoSeqDataLoader(
            dataset_file=f"/home/parthpk/XGeoCLIP/geo-clip_new/data_csv_files/gama_train.csv",
            transform=transform_dict['val'],
        )
    else:
        msls_dataset = GeoSeqDataLoader(
            #dataset_file=f"/home/parthpk/XGeoCLIP/geo-clip_new/data_csv_files/cityguessr_{split}.csv",
            dataset_file=f"/home/parthpk/XGeoCLIP/geo-clip_new/data_csv_files/cityguessr_train_split/cityguessr_train.csv",
            transform=transform_dict[split],
        )

    print(f"Samples in trainset: {len(msls_dataset)}")

    msls_loader = DataLoader(
        dataset=msls_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
    )

    if split in ['val','train_tencrop']:
        n_augs=1
    else:
        n_augs = 5

    for aug in range(n_augs):


        if split in ['train','train_tencrop']:
            Path(f"{CONFIG['output_dir']}/features/{aug}/").mkdir(exist_ok=True, parents=True)
            Path(f"{CONFIG['output_dir']}/labels/{aug}/").mkdir(exist_ok=True, parents=True)

        for batch_num, (video_seq_id, images, labels) in enumerate(tqdm(msls_loader)):
            video_frames = torch.stack(images).squeeze(1)  # Frames, 1, 10, 3, 224, 224

            # T, V, C, H, W = video_frames.shape
            T = video_frames.shape[0]
            C, H, W = video_frames.shape[-3:]
            video_frames = video_frames.view(-1, C, H, W)

            N = video_frames.shape[0]

            chunk_size = 512

            if (N % chunk_size) > 0:
                remainder = 1
            else:
                remainder = 0
            chunks = N // chunk_size + remainder
            # print(chunks)

            try:
                with torch.autocast(device_type="cuda"):
                    with torch.no_grad():
                        if chunks == 1:
                            frame_features = torch.cat([model(video_frames.cuda()).cpu() for model in models], dim=1)
                        else:

                            chunk_feats = []
                            for i in range(chunks-1):
                                start, end = i * chunk_size, (i + 1) * chunk_size
                                # print(start, end)
                                # chunk_feats.append(model(video_frames[start:end].cuda()).cpu())
                                chunk_feats.append(torch.cat([model(video_frames[start:end].cuda()).cpu() for model in models], dim=1))
                                # print(f"step {i}")
                            # chunk_feats.append(model(video_frames[end:].cuda()).cpu())
                            chunk_feats.append(torch.cat([model(video_frames[end:].cuda()).cpu() for model in models], dim=1))
                            frame_features = torch.cat(chunk_feats, dim=0)
                            #print(frame_features.shape)
                            
                            del chunk_feats
                            del video_frames
                            gc.collect()
                            torch.cuda.empty_cache()

            except Exception as e:
                print(video_frames.shape)
                raise e

            if split in ['val', 'train_tencrop']:
                frame_features = frame_features.view(T, 10, -1)
            if split == 'val':
                feat_file = f"{CONFIG['output_dir']}/features/{video_seq_id[0]}.npy"
                target_file = f"{CONFIG['output_dir']}/labels/{video_seq_id[0]}.npy"
            else:
                feat_file = f"{CONFIG['output_dir']}/features/{aug}/{video_seq_id[0]}.npy"
                target_file = f"{CONFIG['output_dir']}/labels/{aug}/{video_seq_id[0]}.npy"
            np.save(feat_file, frame_features.numpy())
            np.save(target_file, labels.cpu().numpy())


if __name__ == "__main__":
    main()
