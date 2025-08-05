import torch
import torch.nn as nn
from transformers import CLIPModel, AutoProcessor
from .tests.transformer_vitbb import Encoder, PositionalEncoder

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='huggingface_hub.*')

class TransformerEmbModel(nn.Module):
    def __init__(self, num_frames_pe = 32, drop_rate_pe = 0, drop_rate = 0.1, in_channels = 1024, cap_scalar = 2, fc_params = [(256, True),(256, True),], embedding_size = 1024, 
        hidden_channels = 1024, embedder_layers = 3, embedder_heads = 8, embedder_d_ff = 1024):
        super().__init__()
        
        self.embedder_layers = embedder_layers
        self.embedding_size = embedding_size

        self.fc_layers = []
        for channels, activate in fc_params:
            channels = channels*cap_scalar
            self.fc_layers.append(nn.Dropout(drop_rate))
            self.fc_layers.append(nn.Linear(in_channels, channels))
            self.fc_layers.append(nn.BatchNorm1d(channels))
            self.fc_layers.append(nn.ReLU(True))
            in_channels = channels
        self.fc_layers = nn.Sequential(*self.fc_layers)
        
        self.video_emb = nn.Linear(in_channels, hidden_channels)
        
        self.video_pos_enc = PositionalEncoder(hidden_channels, drop_rate_pe, seq_len=num_frames_pe)
        if self.embedder_layers > 0:
            self.video_encoder = Encoder(hidden_channels, drop_rate, embedder_heads, embedder_d_ff, self.embedder_layers)
        
        self.embedding_layer = nn.Linear(hidden_channels, embedding_size)

    def forward(self, x, video_masks=None):
        batch_size, num_steps, c = x.shape
        x = x.view(batch_size*num_steps, c)

        # x = self.pooling(x)
        # x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        x = self.video_emb(x)
        x = x.view(batch_size, num_steps, x.size(1))
        x = self.video_pos_enc(x)
        if self.embedder_layers > 0:
            x = self.video_encoder(x, src_mask=video_masks)

        x = x.view(batch_size*num_steps, -1)
        x = self.embedding_layer(x)
        x = x.view(batch_size, num_steps, self.embedding_size)
        return x

class ImageEncoder2(nn.Module):
    def __init__(self, n_frames=1):
        super(ImageEncoder2, self).__init__()
        
        self.n_frames = n_frames

        self.CLIP = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.image_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        if n_frames > 1:
            self.temp_embed = TransformerEmbModel(num_frames_pe = 32, drop_rate_pe = 0, drop_rate = 0.1, in_channels = 768, cap_scalar = 2, fc_params = [(256, True),(256, True),], embedding_size = 768, hidden_channels = 768, embedder_layers = 2, embedder_heads = 8, embedder_d_ff = 768)
        self.mlp = nn.Sequential(nn.Linear(768, 768),
                                 nn.ReLU(),
                                 nn.Linear(768, 512))

        # Freeze CLIP
        for param in self.CLIP.parameters():
            param.requires_grad = False

    def preprocess_image(self, image):
        x = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        return x

    def forward(self, x, precomp=False):
        if not precomp:
            x = self.CLIP.get_image_features(pixel_values=x)
        if self.n_frames > 1:
            x = self.temp_embed(x)
            x = torch.reshape(x, (-1, x.shape[-1]))
        x = self.mlp(x)
        return x

class DINOEncoder2(nn.Module):
    def __init__(self, n_frames=1):
        super(DINOEncoder2, self).__init__()

        self.n_frames = n_frames
        
        self.DINO = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        self.image_processor = AutoProcessor.from_pretrained("facebook/dinov2-large")
        if n_frames > 1:
            self.temp_embed = TransformerEmbModel(num_frames_pe = 32, drop_rate_pe = 0, drop_rate = 0.1, in_channels = 1024, cap_scalar = 2, fc_params = [(256, True),(256, True),], embedding_size = 1024, hidden_channels = 1024, embedder_layers = 2, embedder_heads = 8, embedder_d_ff = 1024)        
        self.mlp = nn.Sequential(nn.Linear(1024, 768),
                                 nn.Mish(),
                                 nn.Linear(768, 512))

        # Freeze DINO
        for param in self.DINO.parameters():
            param.requires_grad = False

    def preprocess_image(self, image):
        x = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        return x

    def forward(self, x, precomp=False):
        if not precomp:
            x = self.DINO(x)
        
        #x = self.mlp(x)
        if self.n_frames > 1:
            x = self.temp_embed(x)
            x = torch.reshape(x, (-1, x.shape[-1]))
        x = self.mlp(x)
        return x

class DINOCLIPEncoder2(nn.Module):
    def __init__(self, n_frames=1, temporal_aggregation=False):
        super(DINOCLIPEncoder2, self).__init__()

        self.n_frames = n_frames
        self.temp_agg = temporal_aggregation

        if temporal_aggregation:
            #self.clip_proj = nn.Sequential(nn.Linear(768, 768))
            #self.dino_proj = nn.Sequential(nn.Linear(1024, 1024))
            if temporal_aggregation == "TempGeo":
                self.temp_embed = TransformerEmbModel(num_frames_pe = 32, drop_rate_pe = 0, drop_rate = 0.1, in_channels = 1792, 
                    cap_scalar = 2, fc_params = [(256, True),(256, True),], embedding_size = 1792, hidden_channels = 1792, 
                    embedder_layers = 2, embedder_heads = 8, embedder_d_ff = 1792)
            elif temporal_aggregation == "TempMLP":
                self.temp_embed = nn.Sequential(nn.Linear(4096, 4096),
                                 nn.Mish(),
                                 nn.Linear(4096, 7168),
                                 nn.Mish())
                # self.temp_embed = nn.Identity()

        self.mlp = nn.Sequential(nn.Linear(1792, 1024),
                                 nn.Mish(),
                                 nn.Linear(1024, 768),
                                 nn.Mish(),
                                 nn.Linear(768, 512))

        # self.mlp_agg = nn.Sequential(nn.Linear(7168, 1024),
        #                              nn.ReLU(),
        #                              nn.Linear(1024, 7168))

    def preprocess_image(self, image):
        x = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        return x

    def forward(self, x, precomp=False):
        if self.n_frames > 1:
            x_clip, x_dino = torch.split(x, [768, 1024], dim=-1)
            # print(x_clip.shape, x_dino.shape)
            #x_clip = self.clip_proj(x_clip)
            #x_dino = self.dino_proj(x_dino)
            #x = torch.cat([x_clip, x_dino], dim=-1)
            if self.temp_agg == "TempGeo":
                x = self.temp_embed(x)
            elif self.temp_agg == "TempMLP":

                # print(x.shape)
                # Project CLIP and DINO features separately into a shared space
                
                # print(x.shape)
                # [B, T, D]
                # pad to multiple of 4
                B, T, D = x.shape
                padding = 4 - (T % 4)
                for _ in range(padding):
                    x = torch.cat([x, x[:,-1:,:]], dim=1)
                _, T_pad, _ = x.shape
                x = x.reshape(B,T_pad//4,-1)
                _, _, D_agg = x.shape
                x = x.reshape(-1, D_agg)
                # print(x.shape)
                # exit()
                x = self.temp_embed(x)
                x = x.reshape(B, T_pad, -1)
                x = x[:,:T,:]
            x = torch.reshape(x, (-1, x.shape[-1]))
        x = self.mlp(x)

        return x

if __name__ == '__main__':
    B, F, D = 128, 8, 1024
    model = DINOEncoder2(n_frames = F)
    inp = torch.randn((B, F, D))
    outp = model(inp, precomp=True)
    print(outp.shape)
