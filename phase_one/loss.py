import torch

def geoclip_loss(logits_img_gps, location_queue, mask_queue, n_views, device):
    # print(logits_img_gps.shape, location_queue.shape, mask_queue.shape, n_views)
    # exit()

    batch_size = logits_img_gps.shape[0]
    crossentropy= torch.nn.CrossEntropyLoss()

    # Define targets
    positives = [1 - j + 2 * i for i in range(batch_size//n_views) for j in range(2)]
    targets_img_gps = torch.Tensor(positives).long().to(device)
    

    # Mask duplicate positive locations
    mask = 1 - torch.eye(batch_size, device=device)  # (2 * batch_size, 2 * batch_size)
    logits_img_gps[:, :mask.shape[1]] = logits_img_gps[:, :mask.shape[1]].masked_fill(mask == 0, float('-inf'))


    # Mask False Negatives
    full_mask = torch.cat([torch.ones((batch_size, batch_size), device=device), mask_queue], dim=1)
    logits_img_gps = logits_img_gps.masked_fill(full_mask == 0, float('-inf'))
    
    img_gps_loss = crossentropy(logits_img_gps, targets_img_gps)

    return img_gps_loss

def full_contrastive_loss(logits_img_gps, location_queue, mask_queue, device):
    # print(logits_img_gps.shape, location_queue.shape, mask_queue.shape)
    # exit()

    batch_size = logits_img_gps.shape[0]
    crossentropy= torch.nn.CrossEntropyLoss()

    # Define targets
    positives = [i for i in range(batch_size)]
    targets_img_gps = torch.Tensor(positives).long().to(device)
    # targets_img_gps = torch.eye(batch_size).long().to(device)


    # Mask False Negatives
    full_mask = torch.cat([torch.ones((batch_size, batch_size), device=device), mask_queue], dim=1)
    logits_img_gps = logits_img_gps.masked_fill(full_mask == 0, float('-inf'))
    
    img_gps_loss = crossentropy(logits_img_gps, targets_img_gps)

    return img_gps_loss

#def temporal_loss(logits_img_gps, n_views, device):
    #write temporal loss

def contrastive_loss(logits_img_gps, device):
    batch_size = logits_img_gps.shape[0]
    crossentropy= torch.nn.CrossEntropyLoss()

    # Define targets
    positives = [i for i in range(batch_size)]
    targets_img_gps = torch.Tensor(positives).long().to(device)
    
    img_gps_loss = crossentropy(logits_img_gps, targets_img_gps)

    return img_gps_loss


