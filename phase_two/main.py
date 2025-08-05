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

import json

from dataloader import (
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
from model import GeoCLIP3
from decoder import XDecoder
from train import train
from eval import eval_images_ten_crop


CONFIG = {
    "mode": "eval",
    "eval_weights" : "1747087654.6666582",#"1729717825.7114346",#"1730166406.1120718", #"1730483118.892898",#"1729457087.909533",#"1714774714.6169744",#"1726600586.5440943",
    "use_temporal": True,
    "use_transformer": True,
    "use_temp_avg": True,
    "use_video_loss": True,
    "lambda_frame":1,
    "trained_name": "temp_tr_avg_lr_5e-5_2layer_dinoclip_16f",#"gama_temp_tr_avg_lr_5e-5_2layer_dinoclip_16f",#"temp_tr_avg_lr_5e-5_2layer_dinoclip_16f",
    "name": "msls_finetune_baseline",
    "vis_bb": "DINOCLIP",
    "loc_pretrain": "geoclip_tr_mply",
    "mlp_pretrain": None,#"geoclip_tr_mply",
    "train_dataset_dir": "../feat_extarct/features/Mapillary/tensors_video_seq_DINOCLIP/train",  #'/home/c3-0/pr288313/datasets/Mapillary/dino_tensors/train',#'./datasets/Mapillary/tensors/train',#'/home/c3-0/pr288313/datasets/Mapillary/dino_tensors/train',##,
    "val_dataset_dir": "../feat_extarct/features/Mapillary/tensors_video_seq_DINOCLIP/val",
    #'val_dataset_dir': '/home/c3-0/pr288313/datasets/Mapillary/tensors_video_seq/val', # '/home/c3-0/pr288313/datasets/Mapillary/dino_tensors/val',#'./datasets/Mapillary/tensors/val',#'/home/c3-0/pr288313/datasets/Mapillary/dino_tensors/val',#
    #'val_dataset_images_dir': '/home/al209167/datasets/im2gps3ktest',
    #'val_dataset_labels': '/home/c3-0/datasets/MP-16/resources/im2gps3k_places365.csv',
    "train_pred_dir": "../predictions_dinoclip_16f_train/1000/3000",
    "val_pred_dir": "/home/parthpk/XGeoCLIP/geo-clip_new/predictions/pred_dump_cvpr25_geo_ft_gallfree/temp_tr_avg_lr_5e-5_2layer_dinoclip_16f/msls_res_0.1/3000",
    "gallery_file": "../data_csv_files/vid_add_0.1_0.005.npy",
    "num_workers": 4,
    "eval_num_workers": 8,
    "num_epochs": 100,
    "dataset_sample": 1.0,
    "train_batch_size": 128,
    'precomp_size':     1,
    "eval_batch_size": 1,
    'train_views': 16,
    "frame_gap": 2,
    "n_encoder" : 1,
    "n_decoder" : 2,
    "lr": 1e-4, 
    "lr_decay": 0.95,
    "warmup_steps": 1000,
    "weight_decay": 0.0,
    "eval_every": 1,
    "mini_val_size": 200,
    "exp_name": "decoder_art_noise_bi_mask_1e-4_16f_weighted_loss_wup_vidloss_2lyr_dec_0.95decay_dinoclip_16fstg1_combo_higher_noise_alpha_0.1",
    "finetune_img": True,
    "finetune_gps": True,
    "seed": 23,
    
}


def main():

    if CONFIG["mode"] == "train":
        CONFIG["start_time"] = str(time.time())

    else:
        CONFIG["start_time"] = CONFIG["eval_weights"]

    CONFIG["results_dir"] = f"../results/{CONFIG['exp_name']}/{CONFIG['start_time']}/"
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
    print(json.dumps(CONFIG))

    ##############################################
    # CREATE MODEL
    ##############################################

    if CONFIG['use_transformer']:
        model = GeoCLIP3(
            from_pretrained=False,
            vision_backbone=CONFIG["vis_bb"],
            gallery_file=CONFIG["gallery_file"],
            n_frames=CONFIG['train_views']
        )
    else:
        model = GeoCLIP3(
            from_pretrained=False,
            vision_backbone=CONFIG["vis_bb"],
            gallery_file=CONFIG["gallery_file"],
            n_frames=1
        )

    decoder = XDecoder(config=CONFIG, device=device)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model = model.to(device)
    decoder = decoder.to(device)
    
    #from fvcore.nn import FlopCountAnalysis, flop_count_table
    
    #flops = FlopCountAnalysis(decoder, (torch.randn(128*16,512).cuda(), torch.randn(128*16,512).cuda()))
    #print(flop_count_table(flops))
    #summary(model.image_encoder.mlp, [[768]])
    #summary(model, input_data=[torch.zeros(1, 3, 224, 224), torch.zeros(1, 2)])
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
            CONFIG["train_pred_dir"],
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
            CONFIG["val_pred_dir"],
            train_or_val="val",
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
        print("Loading model...")
        weights_path = os.path.join(CONFIG["results_dir"], "model.pt")
        ckpt = torch.load(weights_path)
        model.load_state_dict(ckpt, strict=True)

        weights_path = os.path.join(CONFIG["results_dir"], "decoder.pt")
        ckpt = torch.load(weights_path)
        decoder.load_state_dict(ckpt, strict=True)

        accuracy_results, error_results = eval_images_ten_crop(
            msls_dataloader, model, decoder, device, 1000, mini=False, config=CONFIG
        )
        # accuracy_results, error_results = eval_segment(
        #     msls_dataloader, model, device, 1000, mini=False, config=CONFIG
        # )

        exit()

    ##############################################
    # TRAINING
    ##############################################

    print('Loading model...')
    weights_path = os.path.join('../results', CONFIG['trained_name'], CONFIG['eval_weights'], 'model.pt')
    ckpt = torch.load(weights_path)
    model.load_state_dict(ckpt, strict=True)

    
    # if CONFIG["mlp_pretrain"] == "geoclip_tr_mply":
    #     weights_path = os.path.join("../results", CONFIG["trained_name"], "model.pt")
    #     weights_dict = torch.load(weights_path)
    #     mlp_weights = {
    #         k.replace("image_encoder.mlp.", ""): v
    #         for k, v in weights_dict.items()
    #         if "image_encoder.mlp." in k
    #     }
    #     model.image_encoder.mlp.load_state_dict(mlp_weights, strict=True)

    # if CONFIG["loc_pretrain"] == "geoclip_tr_mply":
    #     weights_path = os.path.join("../results", CONFIG["trained_name"], "model.pt")
    #     weights_dict = torch.load(weights_path)
    #     loc_enc_weights = {
    #         k.replace("location_encoder.", ""): v
    #         for k, v in weights_dict.items()
    #         if "location_encoder" in k
    #     }
    #     model.location_encoder.load_state_dict(loc_enc_weights, strict=True)

    # else:
    #     path = f"{file_dir}/weights/location_encoder_weights.pth"
    #     model.location_encoder.load_state_dict(torch.load(path), strict=True)
    #     print("[Location Encoder Init] Weights loaded from ", path)
    

    opt_params = []
    total_param_count = 0
    opt_param_count = 0
    
    loc_param_count = 0
    mlp_param_count = 0
    temp_param_count = 0

    for name, param in model.named_parameters():
        if "DINO" in name:
            continue
        else:
            # print(name)
            total_param_count += param.numel()
        if "location_encoder." in name:
            if CONFIG["finetune_gps"]:
                opt_params.append(param)
                opt_param_count += param.numel()
                loc_param_count += param.numel()
        elif "image_encoder.mlp." in name and CONFIG["finetune_img"]:
            opt_params.append(param)
            opt_param_count += param.numel()
            mlp_param_count += param.numel()
        elif "image_encoder.temp_embed." in name and CONFIG["finetune_img"] and CONFIG["use_transformer"]:
            opt_params.append(param)
            opt_param_count += param.numel()
            temp_param_count += param.numel()
        elif "logit_scale" == name:
            opt_params.append(param)
            opt_param_count += param.numel()
    print(f"{opt_param_count=}, {total_param_count=}")
    print(f"{loc_param_count=}, {mlp_param_count=}, {temp_param_count=}")

    # import sys
    # sys.exit()

    for param in model.parameters():
        param.requires_grad = False
        
    decoder_param_count = 0
    
    for param in decoder.parameters():
        opt_params.append(param)
        opt_param_count += param.numel()
        total_param_count += param.numel()
        decoder_param_count += param.numel()
        param.requires_grad = True

    print(f"{opt_param_count=}, {total_param_count=}")
    print(f"{decoder_param_count=}")

    optimizer = optim.AdamW(
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
        decoder.train()
        model, decoder = train(train_loader, model, decoder, optimizer, epoch, device, scheduler, CONFIG)
        model.eval()
        decoder.eval()

        if epoch % CONFIG["eval_every"] == 0 or epoch > CONFIG["num_epochs"] - 5:
            print("MSLS Ten Crop Evaluation")
            accuracy_results, error_results = eval_images_ten_crop(
                msls_dataloader, model, decoder, device, epoch, mini=True, config=CONFIG
            )
        else:
            accuracy_results[f'acc_1_km'] = 0

        if accuracy_results[f'acc_1_km'] > best_error:
            best_error = accuracy_results[f'acc_1_km']
            best_model_weights = copy.deepcopy(model.state_dict())
            best_decoder_weights = copy.deepcopy(decoder.state_dict())

            weights_path = os.path.join(CONFIG["results_dir"], "model.pt")
            decoder_weights_path = os.path.join(CONFIG["results_dir"], "decoder.pt")
            torch.save(best_model_weights, weights_path)
            torch.save(best_decoder_weights, decoder_weights_path)
            print("*** BEST ERROR")
            print()

    model.load_state_dict(torch.load(weights_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_weights_path, map_location=device))

    print("MSLS Ten Crop Evaluation")
    accuracy_results, error_results = eval_images_ten_crop(
        msls_dataloader, model, decoder, device, epoch, mini=False, config=CONFIG
    )


if __name__ == "__main__":
    main()
