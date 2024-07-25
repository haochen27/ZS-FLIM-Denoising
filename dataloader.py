import torch
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os

def fluore_to_tensor(pic):
    """Convert a ``PIL Image`` to tensor. Range stays the same.
    Only output one channel, if RGB, convert to grayscale as well.
    Currently data is 8 bit depth.
    """
    if not isinstance(pic, Image.Image):
        raise TypeError('pic should be PIL Image. Got {}'.format(type(pic)))

    # Convert PIL Image to numpy array
    img = np.array(pic)
    if pic.mode in ['L', 'P', '1']:
        img = torch.from_numpy(img).unsqueeze(0)
    elif pic.mode in ['RGB', 'RGBA']:
        img = torch.from_numpy(np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])).unsqueeze(0)
    else:
        raise TypeError('Unsupported image type {}'.format(pic.mode))

    return img

def pil_loader(path):
    """Load an image."""
    return Image.open(path).convert('RGB')

class DNdataset(TorchDataset):
    def __init__(self, root, noise_levels, types=None, transform=None, target_transform=None, loader=pil_loader, test_fov=19):
        self.root = root
        self.noise_levels = noise_levels
        self.types = types
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.images = self._make_dataset()
    
    def _make_dataset(self):
        """Scan directories and gather image paths."""
        images = []
        for noise_level in self.noise_levels:
            noise_path = os.path.join(self.root, str(noise_level))
            if self.types:
                for t in self.types:
                    type_path = os.path.join(noise_path, t)
                    for img_name in os.listdir(type_path):
                        img_path = os.path.join(type_path, img_name)
                        images.append(img_path)
            else:
                for img_name in os.listdir(noise_path):
                    img_path = os.path.join(noise_path, img_name)
                    images.append(img_path)
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = self.loader(img_path)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(img)
        else:
            label = img  # Assuming a self-supervised task
        return img, label

def img_loader(root, batch_size, noise_levels, types=None, patch_size=256, test_fov=19):
    """For N2N model, use all captures in each fov, randomly select 2 when loading."""
    transform = transforms.Compose([
        transforms.CenterCrop(patch_size),
        fluore_to_tensor,
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = DNdataset(root, noise_levels, types=types, test_fov=test_fov, transform=transform, target_transform=transform)
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)
    return data_loader
