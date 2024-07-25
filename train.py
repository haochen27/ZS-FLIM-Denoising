import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from model import DNFLIM
from dataloader import DNdataset, generate_data
from torchvision import transforms

def train_model(epochs, batch_size, lr):
    data, labels = generate_data(num_samples=60000, img_size=(256, 256))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = DNdataset(data, labels, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

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

