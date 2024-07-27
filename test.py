import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import DNFLIM
from dataloader import img_loader
from torchvision import transforms

def test_model(batch_size, root, noise_levels, types='Confocal_FISH', pretrained_model='dnflim.pth'):
    
    
    test_loader = img_loader(root, batch_size, noise_levels, types, if_train=False)


    model = DNFLIM()
    if pretrained_model:
        model.load_state_dict(torch.load(pretrained_model))
    criterion = nn.MSELoss()

    total_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            images, _ = data
            outputs = model(images)
            loss = criterion(outputs, images)  # Denoising target is the original input
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f'Average loss on the test dataset: {avg_loss:.4f}')
    return avg_loss

