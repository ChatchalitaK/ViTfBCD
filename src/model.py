"""
ViTfBCD - Vision Transformer for Breast Cancer Detection
Adapted from ViTfSCD (Yang et al., 2023) for histopathological images.
Supports both Binary (Benign/Malignant) and Multi-class (8 subtypes).
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ViT_B_16_Weights, ViT_L_16_Weights


class ClassificationHead(nn.Module):
    """
    Custom classification head replacing ViT's default MLP head.
    Block 4: Flatten -> BatchNorm -> Dense(GeLU) -> BatchNorm -> Softmax
    """
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(x)


class ViTfBCD(nn.Module):
    """
    ViT for Breast Cancer Detection (ViTfBCD)

    Args:
        model_size : 'base' (86M params, 12 layers) or 'large' (307M params, 24 layers)
        num_classes: 2 for binary, 8 for multi-class subtypes
        pretrained  : use ImageNet-21k pretrained weights
        dropout     : dropout rate in classification head
    """
    def __init__(
        self,
        model_size: str = "base",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.model_size = model_size
        self.num_classes = num_classes

        if model_size == "base":
            weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            self.vit = models.vit_b_16(weights=weights)
            self.vit.image_size = 384
            hidden_dim = self.vit.heads.head.in_features  
            
        elif model_size == "large":
            weights = ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None
            self.vit = models.vit_l_16(weights=weights)
            self.vit.image_size = 384  
            hidden_dim = self.vit.heads.head.in_features  
            
        elif model_size == "small":
            weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            self.vit = models.vit_b_16(weights=weights)
            self.vit.image_size = 384 
            self.vit.encoder.layers = self.vit.encoder.layers[:6]
            hidden_dim = self.vit.heads.head.in_features
            
        elif model_size == "tiny":
            weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            self.vit = models.vit_b_16(weights=weights)
            self.vit.image_size = 384 
            self.vit.encoder.layers = self.vit.encoder.layers[:4]
            hidden_dim = self.vit.heads.head.in_features
        else:
            raise ValueError(f"Unknown model_size '{self.model_size}")

        self.vit.heads = ClassificationHead(hidden_dim, num_classes, dropout)

    def resize_position_embeddings(self):
        old_pos_embed = self.vit.encoder.pos_embedding 
    
        class_token_embed = old_pos_embed[:, :1, :]
        patch_embeds = old_pos_embed[:, 1:, :]

        dim = old_pos_embed.shape[-1]
        old_grid_size = int(patch_embeds.shape[1] ** 0.5) 
        new_grid_size = 24 
        
        patch_embeds = patch_embeds.reshape(1, old_grid_size, old_grid_size, dim).permute(0, 3, 1, 2)
        
        new_patch_embeds = torch.nn.functional.interpolate(
            patch_embeds,
            size=(new_grid_size, new_grid_size),
            mode="bilinear",
            align_corners=False
        )
        
        new_patch_embeds = new_patch_embeds.permute(0, 2, 3, 1).reshape(1, -1, dim)

        updated_pos_embed = torch.cat([class_token_embed, new_patch_embeds], dim=1)
        
        self.vit.encoder.pos_embedding = nn.Parameter(updated_pos_embed)
        self.vit.image_size = 384

    def forward(self, x, return_feature: bool = False):
        if not return_feature:
            return self.vit(x)

        x_feat = self.vit._process_input(x)
        n = x_feat.shape[0]
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x_feat = torch.cat([batch_class_token, x_feat], dim=1)
        x_feat = self.vit.encoder(x_feat)

        feature = x_feat[:, 0]
        logits = self.vit.heads(feature)
        return logits, feature

    def get_attention_maps(self, x):
        attn_maps = []

        def hook_fn(module, input, output):
            attn_maps.append(module.attn_weights if hasattr(module, 'attn_weights') else None)

        # Patch: override forward of last encoder block to capture attention
        encoder_blocks = self.vit.encoder.layers
        last_blocks = encoder_blocks[-1]

        attention_weights = []

        def attn_hook(module, input, output):
            pass

        # Use manual forward pass to extract attention
        return self._extract_attention(x)

    def _extract_attention(self, x):

        with torch.no_grad():
            # Process through patch embedding
            x = self.vit._process_input(x)
            n = x.shape[0]

            batch_class_token = self.vit.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

            # Process through encoder layers, capture last layer attention
            attn_weights = None
            for i, block in enumerate(self.vit.encoder.layers):
                normed = block.ln_1(x)
                _, weights = block.self_attention(normed, normed, normed, need_weights=True, average_attn_weights=False)
                if i == len(self.vit.encoder.layers) - 1:
                    attn_weights = weights  
                x = block(x)

            return attn_weights 

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model_size": self.model_size,
            "num_classes": self.num_classes,
            "total_params": f"{total_params / 1e6:.1f}M",
            "trainable_params": f"{trainable_params / 1e6:.1f}M",
        }

    def freeze_backbone(self, num_blocks_to_freeze: int = 8):
        """
        Freezes the patch embedding and the first N blocks of the encoder.
        Helps prevent overfitting on small datasets.
        """
        for param in self.vit.conv_proj.parameters():
            param.requires_grad = False

        self.vit.class_token.requires_grad = False
        self.vit.encoder.pos_embedding.requires_grad = False

        for i, block in enumerate(self.vit.encoder.layers):
            if i < num_blocks_to_freeze:
                for param in block.parameters():
                    param.requires_grad = False
            else:
                for param in block.parameters():
                    param.requires_grad = True

    def get_layered_parameters(self) -> dict:
        """
        Groups trainable parameters into backbone and classification head.
        Returns a dict: {'backbone': list_of_params, 'head': list_of_params}
        """
        backbone_params = []
        head_params = []

        for name, param in self.named_parameters():
            if param.requires_grad:
                if "vit.heads" in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)
                    
        return {"backbone": backbone_params, "head": head_params}


def build_model(config: dict) -> ViTfBCD:
    """Build model from config dictionary."""
    return ViTfBCD(
        model_size=config.get("model_size", "base"),
        num_classes=config.get("num_classes", 2),
        pretrained=config.get("pretrained", True),
        dropout=config.get("dropout", 0.3),
    )