import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from geopy.distance import geodesic as GD
from geopy.distance import distance
from pathlib import Path
import pdb
from torchinfo import summary


def toCartesian(L):
    """ Convert L (lat, lon) to R (x, y, z) """
    L = L * np.pi / 180

    x = torch.cos(L[:, 0]) * torch.cos(L[:, 1])
    y = torch.cos(L[:, 0]) * torch.sin(L[:, 1])
    z = torch.sin(L[:, 0])
    
    R = torch.stack([x, y, z], dim=1)
    return R


def GDkm(L1, L2):
    """ Calculate the distance between two locations in km given L (lat, lon) """
    earth_radius = 6371

    if not isinstance(L1, torch.Tensor): 
        L1 = torch.tensor(L1)
    if not isinstance(L2, torch.Tensor): 
        L2 = torch.tensor(L2)
    if len(L1.shape) == 1:
        L1 = L1.view(1, -1)
    if len(L2.shape) == 1:
        L2 = L2.view(1, -1)

    R1 = toCartesian(L1)
    R2 = toCartesian(L2)

    R1dotR2 = torch.clamp(torch.sum(R1 * R2, dim=1), min=-1, max=1)

    d = torch.acos(R1dotR2) * earth_radius
    return d


def distance_accuracy(targets, preds, distance_thresholds, gps_gallery=None, compute_stats=False):
    total = len(targets)
    correct = {str(dis):0 for dis in distance_thresholds}
    gd_list = []

    for i in range(total):
        # gd = GD(gps_gallery[preds[i]], targets[i]).km
        #gd = GDkm(gps_gallery[preds[i]], targets[i])
        gd = distance(gps_gallery[preds[i]], targets[i]).km
        # pdb.set_trace()
        if compute_stats:
            gd_list.append(gd)
        for dis in distance_thresholds:
            if gd <= dis:
                correct[str(dis)] += 1
        #print(gd, gps_gallery[preds[i]], targets[i], i)
    acc = {k:v/total for k,v in correct.items()}
    if compute_stats:
        gd_list.sort()
        gd_median = gd_list[int(0.5*len(gd_list))]
        gd_25 = gd_list[int(0.25*len(gd_list))]
        gd_75 = gd_list[int(0.75*len(gd_list))]
        gd_90 = gd_list[int(0.9*len(gd_list))]
        gd_10 = gd_list[int(0.1*len(gd_list))]


        gd_avg = sum(gd_list)/total
        return acc, gd_avg, [gd_10, gd_25, gd_median, gd_75, gd_90]
    else:
        return acc

def eval_images(val_dataloader, model, device="cpu"):
    model.eval()
    preds = []
    targets = []

    gps_gallery = model.gps_gallery.to(device)

    with torch.no_grad():
        for imgs, labels in tqdm(val_dataloader, desc="Evaluating"):
            labels = labels.cpu().numpy()
            imgs = imgs.to(device)

            # Get predictions (probabilities for each location based on similarity)
            logits_per_image = model(imgs, gps_gallery)
            probs = logits_per_image.softmax(dim=-1)
            
            # Predict gps location with the highest probability (index)
            outs = torch.argmax(probs, dim=-1).detach().cpu().numpy()
            
            preds.append(outs)
            targets.append(labels)

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    gps_gallery = gps_gallery.cpu().detach().numpy()

    model.train()

    distance_thresholds = [2500, 750, 200, 25, 1] # km
    accuracy_results = {}
    error_results = np.Inf
    print()
    for dis in distance_thresholds:
        acc, avg_distance_error = distance_accuracy(targets, preds, dis, gps_gallery)
        print(f"Accuracy at {dis} km: {acc}, Average Distance Error: {avg_distance_error}")
        accuracy_results[f'acc_{dis}_km'] = acc
    error_results = avg_distance_error
    print()

    return accuracy_results, error_results

def centroid(elements):
    lats = [i[0] for i in elements]
    lons = [i[1] for i in elements]
    return [sum(lats)/len(lats), sum(lons)/len(lons)]
def find_nearest_to_centroid(coordinates):
    """
    Find the index of the coordinate that is closest to the centroid.
    
    Args:
        coordinates: List of tuples containing (latitude, longitude) pairs
        
    Returns:
        int: Index of the coordinate closest to the centroid
        
    Example:
        >>> coords = [(40.7128, -74.0060), (34.0522, -118.2437), (41.8781, -87.6298)]
        >>> find_nearest_to_centroid(coords)
        2  # Index of Chicago coordinates, closest to the centroid
    """
    import math
    
    if not coordinates:
        raise ValueError("Coordinates list cannot be empty")
    
    def calculate_centroid(coords):
        x = y = z = 0
        for lat, lon in coords:
            lat_rad = math.radians(lat)
            lon_rad = math.radians(lon)
            
            x += math.cos(lat_rad) * math.cos(lon_rad)
            y += math.cos(lat_rad) * math.sin(lon_rad)
            z += math.sin(lat_rad)
        
        total_points = len(coords)
        x = x / total_points
        y = y / total_points
        z = z / total_points
        
        central_lon = math.atan2(y, x)
        central_sqrt = math.sqrt(x * x + y * y)
        central_lat = math.atan2(z, central_sqrt)
        
        return (math.degrees(central_lat), math.degrees(central_lon))
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate the great circle distance between two points on Earth."""
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    # Calculate centroid
    centroid_lat, centroid_lon = calculate_centroid(coordinates)
    
    # Find the coordinate closest to the centroid
    min_distance = float('inf')
    nearest_index = 0
    
    for i, (lat, lon) in enumerate(coordinates):
        distance = haversine_distance(centroid_lat, centroid_lon, lat, lon)
        if distance < min_distance:
            min_distance = distance
            nearest_index = i
    
    return nearest_index

def eval_images_ten_crop(val_dataloader, model, device="cpu", epoch=-1, mini=False, config=None):
    model.eval()
    preds = []
    targets = []

    video_preds = []
    video_targets = []

    gps_gallery = model.gps_gallery.to(device)

    i = 0
    
    with torch.no_grad():
        if mini:
            val_len = min(config['mini_val_size'], len(val_dataloader))
        else:
            val_len = len(val_dataloader)

        model.clear_cache()
            
        
        for idx, (seq_id, imgs, labels) in tqdm(enumerate(val_dataloader), desc="Evaluating", total=val_len):
            if idx > config['mini_val_size'] and mini:
                break
            
            imgs = imgs.to(torch.float32)

            labels = labels.cpu().numpy()
            imgs = imgs.to(device)


            #print(imgs.shape)
            #print(labels.shape)



            
            
            # imgs = imgs.squeeze(0)
            # labels = labels.squeeze(0)

            T = labels.shape[1]
            V = imgs.shape[1] // T

            imgs_by_t = imgs.view(T, V, -1) # ????


            imgs_by_v = []
            for aug in range(V):
                imgs_by_v.append(imgs_by_t[:,aug,:])
            imgs_by_v = torch.stack(imgs_by_v)




            # torch.Size([1, 160, 1024]) (1, 16, 2)
            #print(imgs_by_v.shape)
            #exit()
            # Get predictions (probabilities for each location based on similarity)
            if config["use_transformer"]:
                # logits_per_image = model(imgs_by_v, gps_gallery, precomp=True, use_cache=(config['mode'] == 'eval'))
                logits_per_image = model(imgs_by_v, gps_gallery, precomp=True, use_cache=True)
                #logits_per_image = model(imgs_by_v, gps_gallery)
                #if T == 16:
                    #summary(model, input_data=[imgs_by_v, gps_gallery], verbose=1, col_width=16, col_names=["kernel_size", "output_size", "num_params", "mult_adds"], row_settings=["var_names"])
                    #exit()


                if config["city"]:
                    feats = model.embed_feat_img(imgs_by_v).view(V, T, -1).mean(dim=0)
                    folder = f"predictions/features_dump_iccv25_geo_ft_gallfree/{config['exp_name']}/{config['start_time']}/{epoch}/"
                    Path(folder).mkdir(exist_ok=True, parents=True)
                    np.save(f"{folder}/{seq_id}.npy", feats.cpu().numpy())
                    continue
            else:
                logits_per_image = model(imgs_by_v.reshape(-1, imgs_by_v.shape[-1]), gps_gallery, precomp=True, use_cache=True)
            
            logits_per_image = logits_per_image.view(V, T, -1).mean(dim=0)
            # Predict gps location with the highest probability (index)
            outs = torch.argmax(logits_per_image, dim=-1).detach().cpu().numpy()
            
            preds.append(outs)
            targets.append(labels.squeeze(0))

            pred_coords = [gps_gallery[x] for x in outs]
            video_preds.append(outs[find_nearest_to_centroid(pred_coords)])
            video_targets.append(centroid(labels.squeeze(0)))            

            if config['mode'] == "eval":
                outs_10 = torch.argsort(logits_per_image, dim=1, descending=True)[:,:10].detach().cpu().numpy()
                folder = f"predictions/pred_dump_cvpr25_geo_ft_gallfree/{config['exp_name']}/{config['start_time']}/{epoch}/"
                Path(folder).mkdir(exist_ok=True, parents=True)
                np.save(f"{folder}/{seq_id}.npy", gps_gallery[outs_10].cpu().numpy())

            i += 1

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    gps_gallery = gps_gallery.cpu().detach().numpy()

    model.train()

    distance_thresholds = [2500, 750, 200, 25, 5, 1, 0.5, 0.2, 0.1, 0.05, 0.01] # km
    accuracy_results = {}
    error_results = np.Inf
    import time
    results_log_file = open(f"{config['results_dir']}/results-log-{config['exp_name']}.txt", 'a+')
    
    accuracy_results = {}
    acc, avg_distance_error, quartiles = distance_accuracy(targets, preds, distance_thresholds, gps_gallery, compute_stats=True)
    vid_acc, vid_avg_distance_error, vid_quartiles = distance_accuracy(video_targets, video_preds, distance_thresholds, gps_gallery, compute_stats=True)
    print(vid_acc)
    print(vid_avg_distance_error)
    print(vid_quartiles)
    for idx, dis in enumerate(distance_thresholds):

        # if idx == 0:
        #     acc, avg_distance_error, quartiles = distance_accuracy(targets, preds, dis, gps_gallery, compute_stats=True)
        # else:
        #     acc = distance_accuracy(targets, preds, dis, gps_gallery, compute_stats=False)    
        print(f"Accuracy at {dis} km: {acc[str(dis)]}")
        print(f"{epoch},{dis},{acc[str(dis)]}", file=results_log_file,flush=True)
        accuracy_results[f'acc_{dis}_km'] = acc[str(dis)]
    print(f"Average distance error: {avg_distance_error} km")
    print(f"Median distance error: {quartiles[2]} km")
    print(f"{epoch},avg,{avg_distance_error}", file=results_log_file)
    print(f"{epoch},p10,{quartiles[0]}", file=results_log_file)
    print(f"{epoch},p25,{quartiles[1]}", file=results_log_file)
    print(f"{epoch},median,{quartiles[2]}", file=results_log_file)
    print(f"{epoch},p75,{quartiles[3]}", file=results_log_file)
    print(f"{epoch},p90,{quartiles[4]}", file=results_log_file)
    error_results = avg_distance_error
    print()

    return accuracy_results, error_results



def eval_segment(val_dataloader, model, device="cpu", epoch=-1, mini=False, config=None):
    model.eval()
    preds = []
    targets = []

    gps_gallery = model.gps_gallery.to(device)

    i = 0
    frame_count = 0
    with torch.no_grad():
        if mini:
            val_len = config['mini_val_size']
        else:
            val_len = len(val_dataloader)


        gps_features = model.location_encoder(gps_gallery)
        segment_gps_features = []

        for idx, (seq_id, imgs, labels) in tqdm(enumerate(val_dataloader), desc="Evaluating", total=val_len):
            if idx > config['mini_val_size'] and mini:
                break
                
            labels = labels.cpu().numpy()
            T = labels.shape[1]


            # print(imgs.shape)
            # print(labels.shape)
            
            # imgs = imgs.squeeze(0)
            # labels = labels.squeeze(0)






            # torch.Size([1, 160, 1024]) (1, 16, 2)


            # Get predictions (probabilities for each location based on similarity)
            # logits_per_image = model(imgs_by_v, gps_gallery, precomp=True)



            # GPS

            for k in range(T-1):
                segment_gps_feat = torch.cat([gps_features[frame_count + k], gps_features[frame_count + k + 1]], dim=-1)
                segment_gps_features.append(segment_gps_feat)
            segment_gps_last_feat = torch.cat([gps_features[frame_count + T - 1], gps_features[frame_count + T - 1]], dim=-1)
            segment_gps_features.append(segment_gps_last_feat)
            frame_count += T

        segment_gps_features = torch.stack(segment_gps_features, dim=0)
        for idx, (seq_id, imgs, labels) in tqdm(enumerate(val_dataloader), desc="Evaluating", total=val_len):
            labels = labels.cpu().numpy()

            T = labels.shape[1]
            V = imgs.shape[1] // T


            imgs = imgs.to(device)

            imgs_by_t = imgs.view(T, V, -1) # ????


            imgs_by_v = []
            for aug in range(V):
                imgs_by_v.append(imgs_by_t[:,aug,:])
            imgs_by_v = torch.stack(imgs_by_v)


            img_features = model.image_encoder.temp_embed(imgs_by_v)
            # print(img_features.shape)
            img_features = img_features.reshape(V*T, -1)
            # print(img_features.shape)
            img_features = model.image_encoder.mlp(img_features).view(V, T, -1).mean(dim=0)
            # print(img_features.shape)
            # exit()

            video_segment_img_features = torch.cat([img_features[:-1], img_features[1:]],dim=-1)
            video_segment_img_features = torch.cat([video_segment_img_features, torch.cat([img_features[-1].unsqueeze(0), img_features[-1].unsqueeze(0)],dim=-1)], dim=0)


            logits_per_segment = video_segment_img_features @ segment_gps_features.T

            # # print(logits_per_image.shape)
            # logits_per_image = logits_per_image.view(V, T, -1).mean(dim=0)
            
            # # Predict gps location with the highest probability (index)
            outs = torch.argmax(logits_per_segment, dim=-1).detach().cpu().numpy()
            
            preds.append(outs)
            targets.append(labels.squeeze(0))

            # outs_5 = torch.argsort(logits_per_image, dim=1, descending=True)[:,:5].detach().cpu().numpy()

            # np.save(f'predictions_dino/{seq_id}.npy', outs_5)

            # i += 1

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    gps_gallery = gps_gallery.cpu().detach().numpy()

    model.train()

    distance_thresholds = [2500, 750, 200, 25, 5, 1, 0.5, 0.2, 0.1, 0.05, 0.01] # km
    accuracy_results = {}
    error_results = np.Inf
    import time
    results_log_file = open(f"{config['results_dir']}/results-log-{config['exp_name']}.txt", 'a+')
    for idx, dis in enumerate(distance_thresholds):

        if idx == 0:
            acc, avg_distance_error, quartiles = distance_accuracy(targets, preds, dis, gps_gallery, compute_stats=True)
        else:
            acc = distance_accuracy(targets, preds, dis, gps_gallery, compute_stats=False)    
        print(f"Accuracy at {dis} km: {acc}, Average Distance Error: {avg_distance_error}")
        print(f"{epoch},{dis},{acc}", file=results_log_file,flush=True)
        accuracy_results[f'acc_{dis}_km'] = acc
    print(f"{epoch},avg,{avg_distance_error}", file=results_log_file)
    print(f"{epoch},p10,{quartiles[0]}", file=results_log_file)
    print(f"{epoch},p25,{quartiles[1]}", file=results_log_file)
    print(f"{epoch},median,{quartiles[2]}", file=results_log_file)
    print(f"{epoch},p75,{quartiles[3]}", file=results_log_file)
    print(f"{epoch},p90,{quartiles[4]}", file=results_log_file)
    error_results = avg_distance_error
    print()

    return accuracy_results, error_results
