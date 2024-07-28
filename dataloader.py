import torch
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import os


def fluore_to_tensor(pic):
    if not isinstance(pic, Image.Image):
        raise TypeError('pic should be PIL Image. Got {}'.format(type(pic)))

    # Convert PIL Image to numpy array
    img = np.array(pic)
    if pic.mode in ['L', 'P', '1','I', 'F']:
        img = torch.from_numpy(img).unsqueeze(0)
    elif pic.mode in ['RGB', 'RGBA']:
        img = torch.from_numpy(np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])).unsqueeze(0)
    else:
        raise TypeError('Unsupported image type {}'.format(pic.mode))

    return img.float()  

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
        self.train = if_train
        if self.train:
            fovs = list(range(1, 20+1))
            fovs.remove(test_fov)
            self.fovs = fovs
        else:
            self.fovs = [test_fov]
        self.root = root
        self.noise_levels = noise_levels
        self.transform = transform
        self.target_transform = target_transform
        self.images, self.label = self._make_dataset()
    
    def _make_dataset(self):
        images = []
        gt_images = []
        for type in self.types:
            img_path_root = os.path.join(self.root, str(type))
            for noise_level in self.noise_levels:
                if noise_level == int(1):
                    img_path = os.path.join(img_path_root, 'raw')
                else:
                    img_path = os.path.join(img_path_root, 'avg'+str(noise_level))
                gt_path = os.path.join(img_path_root, 'gt')
                if type == 'test_mix':
                    for img_name in sorted(os.listdir(img_path))[:48]:
                        images.append(os.path.join(img_path, img_name))
                    for gt_img in sorted(os.listdir(gt_path))[:48]:
                        gt_images.append(os.path.join((gt_path), gt_img))
                        print(f"Loaded image: {images[-1]}, GT: {gt_images[-1]}")
                    
                else:
                    for fov in self.fovs:
                        current_img_path = os.path.join(img_path, str(fov))
                        gt_img = os.path.join(gt_path, str(fov), 'avg50.png')
                        if self.train:
                            img_name  = random.choice(sorted(os.listdir(current_img_path))[:50])
                            images.append(os.path.join(current_img_path, img_name))
                            gt_images.append(gt_img)
                            print(f"Loaded image: {images[-1]}, GT: {gt_images[-1]}")
                        else:
                            for img_name in sorted(os.listdir(current_img_path))[:50]:
                                images.append(os.path.join(current_img_path, img_name))
                                gt_images.append(gt_img)
                                print(f"Loaded image: {images[-1]}, GT: {gt_images[-1]}")
        return images, gt_images


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
        return img, label, img_path

class FLIM_Dataset(TorchDataset):
    def __init__(self, root, types=None, transform=None, crop_size=256, num_augmentations=50):
        super(FLIM_Dataset, self).__init__()
        self.types = types if types else ['BPAE_Sample1']
        self.transform = transform
        self.crop_size = crop_size
        self.root = root
        self.images_G, self.images_S, self.images = self._make_dataset()
        self.num_augmentations = num_augmentations
    
    def _make_dataset(self):
        imgG_path = []
        imgS_path = []
        imgI_path = []
        for type in self.types:
            img_path_root = os.path.join(self.root, str(type))
            imgG_path.append(os.path.join(img_path_root, 'imageG.tif'))
            imgS_path.append(os.path.join(img_path_root, 'imageS.tif'))
            imgI_path.append(os.path.join(img_path_root, 'imageI.tif'))
        return imgG_path, imgS_path, imgI_path

    @staticmethod
    def img_normalize(image_array):
        min_val = image_array.min()
        max_val = image_array.max()
        if min_val == max_val:
            scaled_array = np.zeros_like(image_array, dtype=np.uint8)
        else:
            scaled_array = (image_array - min_val) / (max_val - min_val)
            scaled_array = scaled_array * 255
        return scaled_array.astype(np.uint8)
    
    def __len__(self):
        return len(self.images) * self.num_augmentations

    def __getitem__(self, idx):
        img_idx = idx // self.num_augmentations
        imageG = Image.open(self.images_G[img_idx])
        imageS = Image.open(self.images_S[img_idx])
        imageA = Image.open(self.images[img_idx])
    
        imageG = np.array(imageG)
        imageS = np.array(imageS)
        imageA = np.array(imageA)

        imageA = self.img_normalize(imageA)
        image_I = imageA * imageG
        image_Q = imageA * imageS

        image_I = Image.fromarray(image_I)
        image_Q = Image.fromarray(image_Q)
        imageA = Image.fromarray(imageA)

        # Get the same crop parameters for all images
        i, j, h, w = transforms.RandomCrop.get_params(imageA, output_size=(self.crop_size, self.crop_size))
        image_I = TF.crop(image_I, i, j, h, w)
        image_Q = TF.crop(image_Q, i, j, h, w)
        imageA = TF.crop(imageA, i, j, h, w)

        if self.transform:
            image_I = self.transform(image_I)
            image_Q = self.transform(image_Q)
            imageA = self.transform(imageA)

        return image_I, image_Q, imageA

def img_loader(root, batch_size, noise_levels, types=None, patch_size=256, test_fov=19, train=True):
    """For N2N model, use all captures in each fov, randomly select 2 when loading."""
    transform = transforms.Compose([
        transforms.CenterCrop(patch_size),
        fluore_to_tensor,
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = DNdataset(root, noise_levels, types=types, test_fov=test_fov, transform=transform, target_transform=transform, if_train=train)
    kwargs = {'num_workers': 4, 'pin_memory': False} if torch.cuda.is_available() else {}
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)
    return data_loader

def img_loader_FLIM(root, batch_size, types=None, patch_size=256, num_augmentations=50):
    transform = transforms.Compose([
        fluore_to_tensor
    ])
    dataset = FLIM_Dataset(root, types=types, transform=transform, num_augmentations=num_augmentations)
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)
    return data_loader