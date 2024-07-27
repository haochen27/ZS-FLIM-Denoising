import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model import DNFLIM
from torchvision import transforms
from dataloader import img_loader

def train_model(epochs, batch_size, lr, root, noise_levels, types):

    train_loader = img_loader(root, batch_size, noise_levels, types)

    model = DNFLIM()
    criterion = nn.MSELoss()  # Assuming denoising task
    optimizer = optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter('runs/DNFLIM_experiment')

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, _) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)  # Denoising target is the original input
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                writer.add_scalar('training loss', running_loss / 100, epoch * len(train_loader) + i)
                running_loss = 0.0

    torch.save(model.state_dict(), 'dnflim.pth')
    writer.close()

