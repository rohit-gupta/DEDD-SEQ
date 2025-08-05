import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .image_encoder2 import ImageEncoder2, DINOEncoder2, DINOCLIPEncoder2
from .location_encoder import LocationEncoder
from .misc import load_gps_data, file_dir

from PIL import Image
from torchvision.transforms import ToPILImage

class GeoCLIP3(nn.Module):
    def __init__(self, from_pretrained=True, vision_backbone='CLIP', queue_size=4096, gallery_file=None, n_frames=1):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if vision_backbone == 'CLIP':
            self.image_encoder = ImageEncoder2(n_frames=n_frames)
        elif vision_backbone == 'DINO':
            self.image_encoder = DINOEncoder2(n_frames=n_frames)
        elif vision_backbone == 'DINOCLIP':
            self.image_encoder = DINOCLIPEncoder2(n_frames=n_frames)
        self.location_encoder = LocationEncoder(from_pretrained=False)

        if gallery_file == None: gallery_file = 'geoclip/model/np_galleries/mp16.npy'
        self.gps_gallery = torch.tensor(np.load(gallery_file), dtype=torch.float32)
        self._initialize_gps_queue(queue_size)

        if from_pretrained:
            self.weights_folder = os.path.join(file_dir, "weights")
            self._load_weights()

        self.device = "cpu"
        self.location_features = None

    def to(self, device):
        self.device = device
        self.image_encoder.to(device)
        self.location_encoder.to(device)
        self.logit_scale.data = self.logit_scale.data.to(device)
        return super().to(device)

    def _load_weights(self):
        self.image_encoder.mlp.load_state_dict(torch.load(f"{self.weights_folder}/image_encoder_mlp_weights.pth"))
        self.location_encoder.load_state_dict(torch.load(f"{self.weights_folder}/location_encoder_weights.pth"))
        self.logit_scale = nn.Parameter(torch.load(f"{self.weights_folder}/logit_scale_weights.pth"))

    def _initialize_gps_queue(self, queue_size):
        self.queue_size = queue_size
        self.register_buffer("gps_queue", torch.randn(2, self.queue_size))
        self.gps_queue = nn.functional.normalize(self.gps_queue, dim=0)
        self.gps_queue *= torch.tensor([[90], [180]])
        self.register_buffer("gps_queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, gps):
        """ Update GPS queue

        Args:
            gps (torch.Tensor): GPS tensor of shape (batch_size, 2)
        """
        gps_batch_size = gps.shape[0]
        gps_ptr = int(self.gps_queue_ptr)
        
        assert self.queue_size % gps_batch_size == 0, f"Queue size {self.queue_size} should be divisible by batch size {gps_batch_size}"

        # Replace the GPS from ptr to ptr+gps_batch_size (dequeue and enqueue)
        self.gps_queue[:, gps_ptr:gps_ptr + gps_batch_size] = gps.t()
        gps_ptr = (gps_ptr + gps_batch_size) % self.queue_size  # move pointer
        self.gps_queue_ptr[0] = gps_ptr

    def get_gps_queue(self):
        return self.gps_queue.t()
                                             
    def forward(self, image, location, precomp=False, use_cache=False):
        """ GeoCLIP's forward pass

        Args:
            image (torch.Tensor): Image tensor of shape (n, 3, 224, 224) or tensor of shape (2n, 768)
            location (torch.Tensor): GPS location tensor of shape (m, 2)
            precomp (bool): flag to ignore the CLIP image encoder ig using precomputed features

        Returns:
            logits_per_image (torch.Tensor): Logits per image of shape (n, m)
        """

        # Compute Features
        image_features = self.image_encoder(image, precomp)
        if use_cache:
            if self.location_features is not None:
                location_features = self.location_features
            else:
                location_features = self.location_encoder(location)
                self.location_features = location_features
        else:
            location_features = self.location_encoder(location)
        logit_scale = self.logit_scale.exp()
        
        # Normalize features
        image_features = F.normalize(image_features, dim=1)
        location_features = F.normalize(location_features, dim=1)
        
        # Cosine similarity (Image Features & Location Features)
        logits_per_image = logit_scale * (image_features @ location_features.t())

        return logits_per_image

    @torch.no_grad()
    def embed_feat_img(self, image, precomp=False):
        return F.normalize(self.image_encoder(image, precomp), dim=1)

    @torch.no_grad()
    def embed_feat_gps(self, location):
        return F.normalize(self.location_encoder(location), dim=1)

    @torch.no_grad()
    def predict(self, image_path, top_k):
        """ Given an image, predict the top k GPS coordinates

        Args:
            image_path (str): Path to the image
            top_k (int): Number of top predictions to return

        Returns:
            top_pred_gps (torch.Tensor): Top k GPS coordinates of shape (k, 2)
            top_pred_prob (torch.Tensor): Top k GPS probabilities of shape (k,)
        """
        image = Image.open(image_path)
        image = self.image_encoder.preprocess_image(image)
        image = image.to(self.device)

        gps_gallery = self.gps_gallery.to(self.device)

        logits_per_image = self.forward(image, gps_gallery)
        probs_per_image = logits_per_image.softmax(dim=-1).cpu()

        # Get top k predictions
        top_pred = torch.topk(probs_per_image, top_k, dim=1)
        top_pred_gps = self.gps_gallery[top_pred.indices[0]]
        top_pred_prob = top_pred.values[0]

        return top_pred_gps, top_pred_prob


class Temporal_Mod(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.encoder = nn.TransformerEncoder(self.layer, num_layers=6)

    def forward(self, image_features):
        return self.encoder(image_features)