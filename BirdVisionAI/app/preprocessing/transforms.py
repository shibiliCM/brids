from torchvision import transforms

def get_transform():
    """Return the image preprocessing pipeline.
    Mirrors the original pipeline: Resize -> CenterCrop -> ToTensor.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
