"""
cnn_model.py - CNN Baseline Models for comparison with ViTfBCD
Model: ResNet50, InceptionV3, EfficientNet-B4, IRv2
All use ImageNet pretrained weight + custom classification head
Same head structure as ViTfBCD for fair comparison

Note: IRv2 uses timm library (pip install timm)
"""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet50_Weights,
    Inception_V3_Weights,
    EfficientNet_B4_Weights, 
)
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("[WARNING] timm not installed. IRv2 unavailable. Run: pip install timm")

class CNNClassificationHead(nn.Module):
    """
    Same head structure as ViTfBCD for fair comparison:
    Linear -> BatchNorm -> GELU -> Dropout -> Linear
    """
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=512), 
            nn.BatchNorm1d(512), 
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.head(x)
    
# ───── ResNet50 ──────────────────────────────────────────────────────────────
class ResNet50Baseline(nn.Module): 
    """
    ResNet50 pretrained on ImageNet, fine-tuned for breast cancer classification.
    Input Size: 384x384
    """
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.model = models.resnet50(weights=weights)

        # Replace final FC layer
        in_features = self.model.fc.in_features
        self.model.fc = CNNClassificationHead(in_features, num_classes, dropout)

    def forward(self, x):
        return self.model(x)
    
    def get_model_info(self): 
        total = sum(p.numel() for p in self.parameters()) 
        return {"name": "ResNet50", "params": f"{total/1e6:.1f}M", "input_size":384}
                
# ───── InceptionV3 ────────────────────────────────────────────────────────────
class InceptionV3Baseline(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.3):
        super().__init__() 
        weight = Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.inception_v3(weights=weight, aux_logits=True)

        # Replace main classifier
        in_features = self.model.fc.in_features
        self.model.fc = CNNClassificationHead(in_features, num_classes, dropout)

        # Replace aux classifier
        in_aux = self.model.AuxLogits.fc.in_features
        self.model.AuxLogits.fc = nn.Linear(in_aux, num_classes)

    def forward(self, x):
        if self.training:
            # return (main_output, aux_output) during training
            output, aux = self.model(x)
            return output, aux
        else:
            return self.model(x)
    
    def get_model_info(self):
        total = sum(p.numel() for p in self.parameters())
        return {"name": "InceptionV3", "params": f"{total/1e6:.1f}M", "input_size": 299} 
    

# ───── EfficientNet-B4 ────────────────────────────────────────────────────────
class EfficientNetB4Baseline(nn.Module):
    """
    EfficientNet-B4 pretrained on ImageNet, fine-tuned for breast cancer classification.
    Input size: 380x380
    """
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.3):
        super().__init__() 
        weight = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.efficientnet_b4(weights=weight)

        # Replace classifier head
        in_features = self.model.classifier[1].in_features 
        self.model.classifier = CNNClassificationHead(in_features, num_classes, dropout)

    def forward(self, x):
        return self.model(x)

    def get_model_info(self):
        total = sum(p.numel() for p in self.parameters())
        return {"name": "EfficientNet-B4", "params": f"{total/1e6:.1f}M", "input_size": 380}

# ───── InceptionResnetV2 (IRv2) ───────────────────────────────────────────────
class IRv2Baseline(nn.Module):
    """
    Inception-ResNet-V2 pretrained on ImageNet via timm.
    Same model family as used in the reference paper (Yang et al., 2023).
    Input size: 299x299
    """
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for IRv2. Run: pip install timm")

        # Load IRv2 from timm with pretrained ImageNet weights
        self.model = timm.create_model(
            "inception_resnet_v2",
            pretrained=pretrained,
            num_classes=0,          
            global_pool="avg",      
        )
        in_features = self.model.num_features   

        # Custom classification head (same as ViTfBCD)
        self.classifier = CNNClassificationHead(in_features, num_classes, dropout)

    def forward(self, x):
        features = self.model(x)    
        return self.classifier(features)

    def get_model_info(self):
        total = sum(p.numel() for p in self.parameters())
        return {"name": "IRv2 (Inception-ResNet-V2)", "params": f"{total/1e6:.1f}M", "input_size": 299}

# ── Factory function ──────────────────────────────────────────────────────────
CNN_MODELS = {
    "resnet50":        ResNet50Baseline,
    "inceptionv3":     InceptionV3Baseline,
    "irv2":            IRv2Baseline,
    "efficientnet_b4": EfficientNetB4Baseline,
}

# Input size each model expects
CNN_INPUT_SIZES = {
    "resnet50":        384,
    "inceptionv3":     299,
    "irv2":            299,
    "efficientnet_b4": 380,
}

def build_cnn_model(name: str, num_classes: int = 8,
                    pretrained: bool = True, dropout: float = 0.3):
    """Build a CNN baseline model by name."""
    name_lower = name.lower() 
    if name_lower not in CNN_MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(CNN_MODELS.keys())}")
    
    model = CNN_MODELS[name_lower](num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    info  = model.get_model_info()
    print(f"[Model] {info['name']} | Params: {info['params']} | Input: {info['input_size']}x{info['input_size']}")
    return model

