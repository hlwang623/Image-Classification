import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _make_transform(input_size, mean=None, std=None):
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD
    return T.Compose([
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


# ========== DINOv2 ==========

def load_dinov2_g(device="cuda"):
    from transformers import AutoModel, AutoImageProcessor
    model_name = "facebook/dinov2-giant"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.eval().to(device)
    model.requires_grad_(False)

    mean = processor.image_mean
    std = processor.image_std
    size = processor.size.get("shortest_edge", processor.size.get("height", 224))
    transform = _make_transform(size, mean=mean, std=std)
    return model, transform, 1536

def _dinov2_forward(model, x):
    outputs = model(x)
    return outputs.last_hidden_state[:, 0, :]


# ========== Phikon-v2 ==========

def load_phikon_v2(device="cuda"):
    from transformers import AutoModel, AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2", trust_remote_code=True)
    model = AutoModel.from_pretrained("owkin/phikon-v2", trust_remote_code=True)
    model = model.eval().to(device)
    model.requires_grad_(False)

    mean = processor.image_mean
    std = processor.image_std
    size = processor.size.get("shortest_edge", 224)
    transform = _make_transform(size, mean=mean, std=std)
    return model, transform, 1024

def _phikon_forward(model, x):
    outputs = model(x)
    return outputs.last_hidden_state[:, 0, :]


# ========== DINOv3 ==========

def load_dinov3_l(device="cuda"):
    from transformers import AutoModel, AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitl14-reg", trust_remote_code=True)
    model = AutoModel.from_pretrained("facebook/dinov3-vitl14-reg", trust_remote_code=True)
    model = model.eval().to(device)
    model.requires_grad_(False)

    mean = processor.image_mean if hasattr(processor, "image_mean") else IMAGENET_MEAN
    std = processor.image_std if hasattr(processor, "image_std") else IMAGENET_STD
    size = 224
    if hasattr(processor, "size"):
        size = processor.size.get("shortest_edge", processor.size.get("height", 224))
    transform = _make_transform(size, mean=mean, std=std)
    return model, transform, 1024

def _dinov3_forward(model, x):
    outputs = model(x)
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state[:, 0, :]
    return outputs.pooler_output


# ========== EVA-02 ==========

def load_eva02_large(device="cuda"):
    import timm
    model = timm.create_model(
        "eva02_large_patch14_clip_224.merged2b_s4b_b131k",
        pretrained=True, num_classes=0
    )
    model = model.eval().to(device)
    model.requires_grad_(False)

    data_cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_cfg, is_training=False)
    feat_dim = model.num_features
    return model, transform, feat_dim

def _timm_forward(model, x):
    return model(x)


# ========== BioMedCLIP ==========

def load_biomedclip(device="cuda"):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    model = model.eval().to(device)
    model.requires_grad_(False)
    return model, preprocess, 512

def _biomedclip_forward(model, x):
    return model.encode_image(x)


# ========== ConvNeXtV2 ==========

def load_convnextv2_large(device="cuda"):
    import timm
    model = timm.create_model(
        "convnextv2_large.fcmae_ft_in22k_in1k_384",
        pretrained=True, num_classes=0
    )
    model = model.eval().to(device)
    model.requires_grad_(False)

    data_cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_cfg, is_training=False)
    feat_dim = model.num_features
    return model, transform, feat_dim


# ========== OpenCLIP ViT-L/14 ==========

def load_openclip_vitl(device="cuda"):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="datacomp_xl_s13b_b90k"
    )
    model = model.eval().to(device)
    model.requires_grad_(False)
    return model, preprocess, 768

def _openclip_forward(model, x):
    return model.encode_image(x)


# ========== Registry ==========

BACKBONE_REGISTRY = {
    "dinov2_g": {
        "loader": load_dinov2_g,
        "forward": _dinov2_forward,
    },
    "phikon_v2": {
        "loader": load_phikon_v2,
        "forward": _phikon_forward,
    },
    "dinov3_l": {
        "loader": load_dinov3_l,
        "forward": _dinov3_forward,
    },
    "eva02_large": {
        "loader": load_eva02_large,
        "forward": _timm_forward,
    },
    "biomedclip": {
        "loader": load_biomedclip,
        "forward": _biomedclip_forward,
    },
    "convnextv2_large": {
        "loader": load_convnextv2_large,
        "forward": _timm_forward,
    },
    "openclip_vitl": {
        "loader": load_openclip_vitl,
        "forward": _openclip_forward,
    },
}


def load_backbone(name, device="cuda"):
    if name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone: {name}. Available: {list(BACKBONE_REGISTRY.keys())}")
    entry = BACKBONE_REGISTRY[name]
    model, transform, feat_dim = entry["loader"](device)
    forward_fn = entry["forward"]
    return model, transform, feat_dim, forward_fn
