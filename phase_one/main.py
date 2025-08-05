import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import numpy as np
import copy
from pathlib import Path
import time

import copy

from geoclip.train.dataloader import (
    GeoDataLoader,
    MapillaryDataset,
    GeoDataLoaderTenCrop,
    GeoDataLoaderPrecomputed,
    MapillaryTenCropDataset,
    MapillaryTemporalDataset,
    img_train_transform,
    img_val_transform_ten_crop,
    temporal_flat_collate_fn,
    temporal_single_collate_fn,
    temporal_collate_fn
)
from geoclip.model import GeoCLIP3
from geoclip.train.train import train, train_temporal
from geoclip.train.eval import eval_images_ten_crop, eval_segment


CONFIG = {
    "mode": "eval",
    "eval_weights" : "1754259568.2614477",#"1729717825.7114346", #"1741387266.9516487",#"1729717825.7114346", # "1713548958.3592417", # "1729717825.7114346",#"1730483118.892898",#"1714774714.6169744",#"1730483118.892898",
    "use_temporal": True,
    "use_transformer": True,
    "use_temp_avg": True,
    "use_agg_mlp": "TempGeo",
    "trained_name": "geoclip_tr_mply_gal_mp16",
    "name": "msls_finetune_baseline",
    "vis_bb": "DINOCLIP",
    "loc_pretrain": "geoclip_tr_mply",
    "mlp_pretrain": None,#"geoclip_tr_mply",
    "train_dataset_dir": "./feat_extarct/features/frank5_full/tensors_video_seq_DINOCLIP/train",  #'/home/c3-0/pr288313/datasets/Mapillary/dino_tensors/train',#'./datasets/Mapillary/tensors/train',#'/home/c3-0/pr288313/datasets/Mapillary/dino_tensors/train',##,
    "val_dataset_dir": "./feat_extarct/features/CityGuessr_feats/tensors_video_seq_DINOCLIP/val",
    #'val_dataset_dir': '/home/c3-0/pr288313/datasets/Mapillary/tensors_video_seq/val', # '/home/c3-0/pr288313/datasets/Mapillary/dino_tensors/val',#'./datasets/Mapillary/tensors/val',#'/home/c3-0/pr288313/datasets/Mapillary/dino_tensors/val',#
    #'val_dataset_images_dir': '/home/al209167/datasets/im2gps3ktest',
    #'val_dataset_labels': '/home/c3-0/datasets/MP-16/resources/im2gps3k_places365.csv',
    #"gallery_file": "data_csv_files/vid_full.npy",#vid_add_0.1_0.005.npy
    "gallery_file": "data_csv_files/cityguessr_full.npy",#gama_add_0.5_0.npy",
    "city": None, # eval only argument
    "num_workers": 4,
    "eval_num_workers": 8,
    "num_epochs": 200,
    "dataset_sample": 1.0,
    "train_batch_size": 128,
    'precomp_size':     1,
    "eval_batch_size": 1,
    'train_views': 16,
    "frame_gap": 3,
    "lr": 5e-5,
    "lr_decay": 0.97,
    "weight_decay": 0.0,
    "eval_every": 5,
    "mini_val_size": 1000,
    "exp_name": "temp_tr_avg_lr_5e-5_2layer_dinoclip_16f_frank5_full",#"gama_temp_tr_avg_lr_5e-5_2layer_dinoclip_16f",#"temp_tr_avg_lr_5e-5_2layer_dinoclip_16f",
    "finetune_img": True,
    "finetune_gps": True,
    "seed": 23,
}


def main():

    if CONFIG["mode"] == "train":
        CONFIG["start_time"] = str(time.time())

    else:
        CONFIG["start_time"] = CONFIG["eval_weights"]

    CONFIG["results_dir"] = f"results/{CONFIG['exp_name']}/{CONFIG['start_time']}/"
    Path(CONFIG["results_dir"]).mkdir(exist_ok=True, parents=True)

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

    ##############################################
    # CREATE MODEL
    ##############################################

    if CONFIG['use_transformer']:
        model = GeoCLIP3(
            from_pretrained=False,
            vision_backbone=CONFIG["vis_bb"],
            temporal_aggregation=CONFIG["use_agg_mlp"],
            gallery_file=CONFIG["gallery_file"],
            n_frames=CONFIG['train_views'],

        )
    else:
        model = GeoCLIP3(
            from_pretrained=False,
            vision_backbone=CONFIG["vis_bb"],
            gallery_file=CONFIG["gallery_file"],
            n_frames=1
        )

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model = model.to(device)
    
    #from fvcore.nn import FlopCountAnalysis

    #inputs = torch.randn(128, 16, 1792)  # Example input shape
    #inputs = inputs.to(device)
    #flops = FlopCountAnalysis(model.image_encoder, inputs)
    #print(flops.total())  # Total FLOPs
    #print(flops.total() / 1e9, "GFLOPs")
    
    #exit()

    ##############################################
    # DATALOADER
    ##############################################

    train_transform = img_train_transform()
    transform = img_val_transform_ten_crop()

    """
    train_dataset = GeoDataLoaderPrecomputed(
        dataset_dir=CONFIG['train_dataset_dir'],
        dataset_sample=CONFIG['dataset_sample'],
        seed=CONFIG['seed'],
    )
    """

    if CONFIG["use_temporal"]:
        train_dataset = MapillaryTemporalDataset(
            CONFIG["train_dataset_dir"],
            train_or_val="train",
            n_frames=CONFIG["train_views"],
            gap=CONFIG["frame_gap"],
            dataset_sample=1.0,
            seed=23,
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=CONFIG["train_batch_size"],
            shuffle=True,
            num_workers=CONFIG["num_workers"],
            collate_fn=temporal_collate_fn,
            drop_last=True
        )
    else:

        train_dataset = MapillaryDataset(
            dataset_dir=CONFIG["train_dataset_dir"], n_views=CONFIG["train_views"]
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=CONFIG["train_batch_size"],
            shuffle=True,
            num_workers=CONFIG["num_workers"],
        )

   

    # msls_dataset = GeoDataLoaderTenCrop(
    #     dataset_file='/home/c3-0/parthpk/Mapilliary_data/mapilliary_val_seq.csv',
    #     dataset_folder='',
    #     transform=transform
    # )

    if CONFIG["use_temporal"]:
        msls_dataset = MapillaryTemporalDataset(
            CONFIG["val_dataset_dir"],
            train_or_val="val",
            city=CONFIG["city"],
            n_frames=None,
            dataset_sample=1.0,
            seed=23,
        )

        msls_dataloader = DataLoader(
            dataset=msls_dataset,
            batch_size=CONFIG["eval_batch_size"],
            shuffle=True,
            num_workers=CONFIG["eval_num_workers"],
            collate_fn=temporal_single_collate_fn,
        )

    else:
        msls_dataset = MapillaryTenCropDataset(
            dataset_dir=CONFIG["val_dataset_dir"],
        )

        msls_dataloader = DataLoader(
            dataset=msls_dataset,
            batch_size=CONFIG["eval_batch_size"],
            shuffle=False,
            num_workers=CONFIG["eval_num_workers"],
        )

    ##############################################
    # EVAL ONLY
    ##############################################

    if CONFIG["mode"] == "eval":
        print(CONFIG["gallery_file"])
        print("Loading model...")
        weights_path = os.path.join(CONFIG["results_dir"], "model.pt")
        ckpt = torch.load(weights_path)
        model.load_state_dict(ckpt, strict=True)

        accuracy_results, error_results = eval_images_ten_crop(
            msls_dataloader, model, device, 3000, mini=False, config=CONFIG
        )
        # accuracy_results, error_results = eval_segment(
        #     msls_dataloader, model, device, 1000, mini=False, config=CONFIG
        # )

        exit()

    ##############################################
    # TRAINING
    ##############################################

    # print('Loading model...')
    # weights_path = os.path.join('results', CONFIG['trained_name'], 'model.pt')
    # ckpt = torch.load(weights_path)
    # model.load_state_dict(ckpt, strict=True)

    
    if CONFIG["mlp_pretrain"] == "geoclip_tr_mply":
        weights_path = os.path.join("results", CONFIG["trained_name"], "model.pt")
        weights_dict = torch.load(weights_path)
        mlp_weights = {
            k.replace("image_encoder.mlp.", ""): v
            for k, v in weights_dict.items()
            if "image_encoder.mlp." in k
        }
        model.image_encoder.mlp.load_state_dict(mlp_weights, strict=True)

    if CONFIG["loc_pretrain"] == "geoclip_tr_mply":
        weights_path = os.path.join("results", CONFIG["trained_name"], "model.pt")
        weights_dict = torch.load(weights_path)
        loc_enc_weights = {
            k.replace("location_encoder.", ""): v
            for k, v in weights_dict.items()
            if "location_encoder" in k
        }
        model.location_encoder.load_state_dict(loc_enc_weights, strict=True)

    else:
        path = f"{file_dir}/weights/location_encoder_weights.pth"
        model.location_encoder.load_state_dict(torch.load(path), strict=True)
        print("[Location Encoder Init] Weights loaded from ", path)
    

    opt_params = []
    total_param_count = 0
    opt_param_count = 0

    for name, param in model.named_parameters():
        if "CLIP" in name:
            continue
        else:
            # print(name)
            total_param_count += param.numel()
        if "location_encoder." in name:
            if CONFIG["finetune_gps"]:
                opt_params.append(param)
                opt_param_count += param.numel()
        elif "image_encoder.mlp." in name and CONFIG["finetune_img"]:
            opt_params.append(param)
            opt_param_count += param.numel()
        elif "image_encoder.temp_embed." in name and CONFIG["finetune_img"] and CONFIG["use_transformer"]:
            opt_params.append(param)
            opt_param_count += param.numel()
        elif "logit_scale" == name:
            opt_params.append(param)
            opt_param_count += param.numel()
    print(f"{opt_param_count=}, {total_param_count=}")
    # import sys
    # sys.exit()

    optimizer = optim.Adam(
        opt_params, lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=CONFIG["lr_decay"]
    )
    # criterion = nn.CrossEntropyLoss().to(device)

    best_error = 0
    # best_model_weights = None
    for epoch in range(CONFIG["num_epochs"]):
        model.train()
        model = train(train_loader, model, optimizer, epoch, device, scheduler, CONFIG)
        model.eval()

        if epoch % CONFIG["eval_every"] == 0 or epoch > CONFIG["num_epochs"] - 5:
            print("MSLS Ten Crop Evaluation")
            accuracy_results, error_results = eval_images_ten_crop(
                msls_dataloader, model, device, epoch, mini=True, config=CONFIG
            )
        else:
            accuracy_results[f'acc_1_km'] = 0

        if accuracy_results[f'acc_1_km'] > best_error:
            best_error = accuracy_results[f'acc_1_km']
            best_model_weights = copy.deepcopy(model.state_dict())

            weights_path = os.path.join(CONFIG["results_dir"], "model.pt")
            torch.save(best_model_weights, weights_path)
            print("*** BEST ERROR")
            print()

    model.load_state_dict(torch.load(weights_path, map_location=device))

    print("MSLS Ten Crop Evaluation")
    accuracy_results, error_results = eval_images_ten_crop(
        msls_dataloader, model, device, epoch, mini=False, config=CONFIG
    )


if __name__ == "__main__":
    main()
