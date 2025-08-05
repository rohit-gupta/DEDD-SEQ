import torch
import torch.nn as nn
import clip
from transformers import SiglipVisionModel

class CLIPVisualEncoder(nn.Module):
    def __init__(self, architecture, freeze_weights=True):
        """
        architecture: str (ViT-B/32, ViT-B/16, ViT-L/14)
        """
        super().__init__()

        self.freeze_weights = freeze_weights

        # Load CLIP and freeze parameters
        self.clip, _ = clip.load(architecture, "cpu")
        if self.freeze_weights:
            for param in self.clip.parameters():
                param.requires_grad = False

    def forward(self, x):
        with torch.set_grad_enabled(not self.freeze_weights):
            x = self.clip.encode_image(x).float()
        return x
        
class SiglipMLVisualEncoder(nn.Module):
    def __init__(self, freeze_weights=True):
        """
        architecture: str (ViT-B/32, ViT-B/16, ViT-L/14)
        """
        super().__init__()

        self.freeze_weights = freeze_weights

        # Load CLIP and freeze parameters
        self.siglipml = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-256-multilingual")
        self.embeddings = self.siglipml.vision_model
        if self.freeze_weights:
            for param in self.siglipml.parameters():
                param.requires_grad = False

    def forward(self, x):
        with torch.set_grad_enabled(not self.freeze_weights):
            x = self.embeddings(x).pooler_output
        return x


class ImageEncoder(nn.Module):
    def __init__(self, architecture="ViT-L/14", freeze_backbone=True):
        super().__init__()

        self.clip_visual_encoder = CLIPVisualEncoder(architecture, freeze_backbone)

        self.fc = nn.Sequential(nn.Linear(768, 768), nn.ReLU())

        self.proj = nn.Linear(768, 512)

    def forward(self, x):
        x = self.clip_visual_encoder(x)
        x = self.fc(x)
        return self.proj(x)
        
        
if __name__ == "__main__":
    model = SiglipMLVisualEncoder()

    x = torch.rand(4, 3, 256, 256)  # shape (batch_size, channels, height, width)
        
    model, x = model.cuda(), x.cuda()
        
    output = model(x)
        
    print("Shapes of the outputs:")
    print(f"Output: {output.shape}")
