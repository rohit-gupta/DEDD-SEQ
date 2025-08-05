import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import math

import pdb
from loss import geoclip_loss, full_contrastive_loss, contrastive_loss


def addGaussianNoise(L, std_meters):
    """ Add Gaussian noise to the coordinates """
    noise = torch.normal(0, std_meters, size=L.shape).to(L.device)

    # Scale the noise to lat, lon
    lat_noise = noise[:, 0] / 111132.954
    lon_noise = noise[:, 1] / (111132.954 * torch.cos(L[:, 0] * math.pi / 180))

    # Add the noise to the coordinates
    L_noise = L + torch.stack([lat_noise, lon_noise], dim=1)

    return L_noise


def getGPSQueueFeatures(gps, gps_features, model, noise=1000):
    # Get the queues
    location_queue = model.gps_queue.t().detach()

    # Add noise to the queue
    #location_queue = addGaussianNoise(location_queue, noise)

    # Get the queue features
    gps_queue_features = model.location_encoder(location_queue)

    # Normalize the queue features
    gps_queue_features = F.normalize(gps_queue_features, dim=1)

    # Concatenate Features
    gps_features = torch.cat([gps_features, gps_queue_features], dim=0)

    # Add GPS to Queue
    model.dequeue_and_enqueue(gps)

    return gps_features, location_queue


def toCartesian(L):
    """ Convert L (lat, lon) to R (x, y, z) """
    L = L * math.pi / 180

    x = torch.cos(L[:, 0]) * torch.cos(L[:, 1])
    y = torch.cos(L[:, 0]) * torch.sin(L[:, 1])
    z = torch.sin(L[:, 0])
    
    R = torch.stack([x, y, z], dim=1)
    return R


def pairwise_distance_matrix(L1, L2):
    """ Calculate the pairwise distance matrix for L (lat, lon) """
    # Step 1: Convert GPS coordinates to Cartesian coordinates
    R1 = toCartesian(L1)
    R2 = toCartesian(L2)
    
    # Step 2: Calculate pairwise distance between every pair of Cartesian coordinates
    cosine = torch.einsum('ij,kj->ik', R1, R2)  # calculate cosine similarity
    
    # Clamp the values to be between -1 and 1 to avoid NaNs in acos
    cosine = torch.clamp(cosine, -1, 1)

    # Convert from cosine distance to geodesic distance
    d = torch.acos(cosine) * 6371  # multiply by Earth's radius to get distance in km

    # Step 3: Return the resulting distance matrix
    return d


def distance_mask(L1, L2, km = 1.0):
    """ Create a mask for the distance matrix """
    # Step 1: Calculate the pairwise distance matrix
    d = pairwise_distance_matrix(L1, L2)

    # Step 2: Create a mask for the distance matrix
    mask = (d > km).float()

    # Step 3: Return the mask
    return mask


def train(train_dataloader, model, decoder, optimizer, epoch, device, scheduler=None, config=None):
    print("Starting Epoch", epoch)

    num_steps = len(train_dataloader)

    bar = tqdm(enumerate(train_dataloader), total=num_steps)

    # targets_img_gps = torch.Tensor([1 - j + 2 * i for i in range(batch_size) for j in range(2)]).long().to(device)
    # mask = 1 - torch.eye(2 * batch_size).to(device) # (2 * batch_size, 2 * batch_size)
    
    import time
    train_log_file = open(f"{config['results_dir']}/train-log-{config['exp_name']}.txt", 'a+')

    for i ,(img_features, pred_gps, gps) in bar:

        steps = i + epoch * num_steps 

        if steps < config["warmup_steps"]:
            fraction = float(steps+1)/config["warmup_steps"]

            for g in optimizer.param_groups:
                g['lr'] = fraction * config["lr"]


        # print(img_features.shape, gps.shape, batch_size)

        if not config["use_temporal"]:
            img_features = img_features.squeeze(0)
        batch_size = img_features.shape[0]

        img_features = img_features.to(device) # (1, 2 * batch_size, 768)
        pred_gps = pred_gps.to(device)
        gps = gps.to(device)

        # for i in range(len(gps)):
        #     print(gps[i], pred_gps[i])
        # exit()

        optimizer.zero_grad()

        B,T,D = img_features.shape

        pred_gps = pred_gps.view(-1, 2)
        gps = gps.view(-1, 2)

        if config["use_transformer"]:
            img_features = model.image_encoder.temp_embed(img_features)
        img_features = img_features.view(B*T, -1)
        img_features = model.image_encoder.mlp(img_features)

        pred_gps_features = model.location_encoder(pred_gps)
        gps_features = model.location_encoder(gps)
        #gps_features = gps_features.reshape(B, T, -1)

        # Normalize the features
        img_features = F.normalize(img_features, dim=1)
        pred_gps_features = F.normalize(pred_gps_features, dim=1)
        gps_features = F.normalize(gps_features, dim=1)
        
        new_gps_features = decoder(img_features, pred_gps_features)
        new_gps_features = F.normalize(new_gps_features, dim=1)

        temp = model.logit_scale.exp()

        # Get the logits
        #logits_img_gps = temp * (new_gps_features @ gps_features.T)
        dist = (new_gps_features @ gps_features.T)
        target = torch.eye(dist.shape[0]).to(device)
        #print(torch.triu(dist).mean(), torch.tril(dist).mean(), torch.diagonal(dist).mean())
        loss_matrix = F.mse_loss(dist, target, reduction='none')
        #print(torch.triu(loss_matrix).mean(), torch.tril(loss_matrix).mean(), torch.diagonal(loss_matrix).mean())

        l_pos = 1.
        l_neg = 0.01
        
        # frame-wise loss
        frame_loss = l_neg*torch.triu(loss_matrix).mean() + l_neg*torch.tril(loss_matrix).mean() + l_pos*torch.diagonal(loss_matrix).mean()


        if config["use_video_loss"]:
        
            new_gps_features_mean = new_gps_features.reshape(config["train_batch_size"], config["train_views"], -1).mean(dim=1)
            gps_features_mean = gps_features.reshape(config["train_batch_size"], config["train_views"], -1).mean(dim=1)

            vid_dist = (new_gps_features_mean @ gps_features_mean.T)
            vid_target = torch.eye(vid_dist.shape[0]).to(device)
            vid_loss_matrix = F.mse_loss(vid_dist, vid_target, reduction='none')

            # video-wise loss
            vid_loss = l_neg*torch.triu(vid_loss_matrix).mean() + l_neg*torch.tril(vid_loss_matrix).mean() + l_pos*torch.diagonal(vid_loss_matrix).mean()

        
        loss = config['lambda_frame']*frame_loss + vid_loss

        # loss = geoclip_loss(logits_img_gps, location_queue, queue_mask, config['train_views'], device)

        #loss = contrastive_loss(logits_img_gps, device)




        # Backpropagate
        loss.backward()
        optimizer.step()
        
        print(f"{epoch},{i},{loss.item()}",file=train_log_file,flush=True)

        bar.set_description("Epoch {} loss: {:.5f}".format(epoch, loss.item()))

    if scheduler is not None:
        scheduler.step()

    return model, decoder

def train_temporal(train_dataloader, model, temporal_model, optimizer, epoch, device, scheduler=None, config=None):
    print("Starting Epoch", epoch)

    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

    # targets_img_gps = torch.Tensor([1 - j + 2 * i for i in range(batch_size) for j in range(2)]).long().to(device)
    # mask = 1 - torch.eye(2 * batch_size).to(device) # (2 * batch_size, 2 * batch_size)
    
    import time
    train_log_file = open(f"{config['results_dir']}/train-log-{config['exp_name']}.txt", 'a+')

    for i ,(img_features, gps) in bar:

        batch_size = img_features.shape[0]

        if batch_size != config['train_views'] * config['precomp_size'] * config['train_batch_size']:
            continue
        # print(img_features.shape, gps.shape)
        img_features = img_features.to(device) # (1, 2 * batch_size, 768)
         # (2 * batch_size, 768)

        gps = gps.to(device)

        optimizer.zero_grad()

        img_features = model.image_encoder.mlp(img_features)
        gps_features = model.location_encoder(gps)

        # Normalize the features
        img_features = F.normalize(img_features, dim=1)
        gps_features = F.normalize(gps_features, dim=1)

        temporal_img_features = temporal_model(img_features)

        # Get the temperature
        temp = model.logit_scale.exp()

        # Get the logits
        logits_img_gps = temp * (img_features @ gps_features.T)

        loss = temporal_loss(logits_img_gps, config['train_views'], device)


        # Backpropagate
        loss.backward()
        optimizer.step()
        
        print(f"{epoch},{i},{loss.item()}",file=train_log_file,flush=True)

        bar.set_description("Epoch {} loss: {:.5f}".format(epoch, loss.item()))

    if scheduler is not None:
        scheduler.step()

    return model
