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
    def __init__(self, root, noise_levels, types=None, transform=None, target_transform=None,test_fov=19,if_train=True):
        super(DNdataset, self).__init__()
        all_noise_levels = [1, 2, 4, 8, 16]
        all_types = ['TwoPhoton_BPAE_R', 'TwoPhoton_BPAE_G', 'TwoPhoton_BPAE_B',
            'TwoPhoton_MICE', 'Confocal_MICE', 'Confocal_BPAE_R',
            'Confocal_BPAE_G', 'Confocal_BPAE_B', 'Confocal_FISH',
            'WideField_BPAE_R', 'WideField_BPAE_G', 'WideField_BPAE_B']
        assert all([level in all_noise_levels for level in noise_levels])
        if types is None:
            self.types = all_types
        else:      
            self.types = types
        self.root = root
        self.noise_levels = noise_levels
        self.transform = transform
        self.target_transform = target_transform
        self.images, self.label = self._make_dataset()
        self.train = if_train
        self.fovs = [test_fov]
        
        # if self.train:
        #     fovs = list(range(1, 20+1))
        #     fovs.remove(test_fov)
        #     self.fovs = fovs
        # else:
        #     self.fovs = [test_fov]
    
    def _make_dataset(self):
        """Scan directories and gather image paths."""
        images = []
        gt_images = []
        for type in self.noise_levels:
            img_path = os.path.join(self.root, str(type))
            for noise_level in self.types:
                if noise_level == 1:
                    img_path = os.path.join(img_path, 'raw')
                else:
                    img_path = os.path.join(img_path, 'avg'+str(noise_level))
                    gt_path = os.path.join(img_path, 'gt')
                for fov in self.fovs:
                    img_path = os.path.join(img_path, str(fov))
                    gt_img = os.path.join(gt_path, str(fov),'avg50.png')
                    if self.train:
                        for img_name in sorted(os.listdir(img_path))[:50]:
                            images.append(os.path.join(img_path, img_name))
                            gt_images.append(gt_img)
        return images,gt_images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        label_path = self.label[idx]
        label = Image.open(label_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
            label = self.target_transform(label)
        return img, label

def img_loader(root, batch_size, noise_levels, types=None, patch_size=256, test_fov=19, if_train=True):
    """For N2N model, use all captures in each fov, randomly select 2 when loading."""
    transform = transforms.Compose([
        transforms.CenterCrop(patch_size),
        fluore_to_tensor,
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = DNdataset(root, noise_levels, types=types, test_fov=test_fov, transform=transform, target_transform=transform, if_train=if_train)
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)
    return data_loader
