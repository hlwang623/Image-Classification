import torch


D4_TRANSFORMS = {
    "identity": lambda x: x,
    "hflip": lambda x: x.flip(-1),
    "vflip": lambda x: x.flip(-2),
    "rot180": lambda x: x.flip(-1).flip(-2),
    "rot90": lambda x: x.transpose(-2, -1).flip(-1),
    "rot270": lambda x: x.transpose(-2, -1).flip(-2),
    "transpose": lambda x: x.transpose(-2, -1),
    "anti_transpose": lambda x: x.flip(-1).flip(-2).transpose(-2, -1),
}

TTA_PRESETS = {
    "fast": ["identity", "hflip", "vflip"],
    "safe": list(D4_TRANSFORMS.keys()),
    "full": list(D4_TRANSFORMS.keys()),
}


def get_tta_transforms(mode="safe"):
    names = TTA_PRESETS.get(mode, TTA_PRESETS["safe"])
    return [D4_TRANSFORMS[n] for n in names]
