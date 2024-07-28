import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from model import N2N_Autoencoder, UNet_SharedEncoder
from dataloader import img_loader,img_loader_FLIM
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np

def train_model(epochs, batch_size, lr, root, noise_levels, types , alpha=0.8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = img_loader(root, batch_size, noise_levels, types)
    eval_loader = img_loader(root, batch_size, noise_levels, types, train=False)

    model = N2N_Autoencoder(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load('./model/best_dnflim.pth'))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f'runs/DNFLIM_experiment_{timestamp}')

    psnr_metric = PeakSignalNoiseRatio().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure().to(device)

    best_eval_loss = float('inf')
    prev_loss = None
    unstable_epochs = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_psnr = 0.0
        running_ssim = 0.0
        num_batches = 0

        for inputs, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = alpha * criterion(outputs, labels) + (1-alpha) * (calculate_entropy(abs(outputs -labels)))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_psnr += psnr_metric(outputs, labels).item()
            running_ssim += ssim_metric(outputs, labels).item()
            num_batches += 1

        avg_epoch_loss = running_loss / num_batches
        avg_epoch_psnr = running_psnr / num_batches
        avg_epoch_ssim = running_ssim / num_batches
        writer.add_scalar('epoch_loss', avg_epoch_loss, epoch)
        writer.add_scalar('epoch_psnr', avg_epoch_psnr, epoch)
        writer.add_scalar('epoch_ssim', avg_epoch_ssim, epoch)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.4f}, PSNR: {avg_epoch_psnr:.4f}, SSIM: {avg_epoch_ssim:.4f}')

        if prev_loss is not None:
            loss_change = (avg_epoch_loss - prev_loss) / abs(prev_loss)
            if loss_change < 0.15 and loss_change > -1:
                unstable_epochs += 1
                if unstable_epochs >= 3:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.2
                    print(f"Reduced learning rate to {optimizer.param_groups[0]['lr']}")
                    unstable_epochs = 0
        prev_loss = avg_epoch_loss

        model.eval()
        eval_loss = 0.0
        eval_psnr = 0.0
        eval_ssim = 0.0
        eval_batches = 0

        with torch.no_grad():
            for eval_inputs, eval_labels, _ in eval_loader:
                eval_inputs, eval_labels = eval_inputs.to(device), eval_labels.to(device)
                eval_outputs = model(eval_inputs)
                eval_loss += criterion(eval_outputs, eval_labels).item()
                eval_psnr += psnr_metric(eval_outputs, eval_labels).item()
                eval_ssim += ssim_metric(eval_outputs, eval_labels).item()
                eval_batches += 1

        avg_eval_loss = eval_loss / eval_batches
        avg_eval_psnr = eval_psnr / eval_batches
        avg_eval_ssim = eval_ssim / eval_batches
        writer.add_scalar('eval loss', avg_eval_loss, epoch)
        writer.add_scalar('eval psnr', avg_eval_psnr, epoch)
        writer.add_scalar('eval ssim', avg_eval_ssim, epoch)
        print(f'Epoch [{epoch + 1}/{epochs}], Eval Loss: {avg_eval_loss:.4f}, Eval PSNR: {avg_eval_psnr:.4f}, Eval SSIM: {avg_eval_ssim:.4f}')

        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            torch.save(model.state_dict(), './model/best_dnflim.pth')
            print(f'Saved Best Model at Epoch {epoch + 1} with Eval Loss: {avg_eval_loss:.4f}')

    torch.save(model.state_dict(), './model/dnflim.pth')
    writer.close()

def ZS_FLIM_train_model(epochs, batch_size, lr, root, types, alpha=0.8):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = img_loader_FLIM(root, batch_size, types)

    model_FLIM = UNet_SharedEncoder(in_channels=1, out_channels=1).to(device)
    model_intensity = N2N_Autoencoder(in_channels=1, out_channels=1).to(device)
    model_intensity.load_state_dict(torch.load('./model/best_dnflim.pth'))
    model_intensity.eval()
    optimizer = optim.Adam(model_FLIM.parameters(), lr=lr)
    creterion = nn.MSELoss()
    psnr_metric = PeakSignalNoiseRatio().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    for epoch in range(epochs):
        model_FLIM.train()
        running_loss = 0.0
        running_psnr = 0.0
        running_ssim = 0.0
        num_batches = 0

        for imageI, imageQ, imageA in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"):
            imageI, imageQ, imageA = imageI.to(device), imageQ.to(device), imageA.to(device)
            optimizer.zero_grad()

            outputs_intensity = model_intensity(imageA)
            output_Q,output_I = model_FLIM(imageQ,imageI)   
            
            
            loss = alpha*creterion((output_Q**2+output_I**2), outputs_intensity*output_I)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = running_loss / num_batches

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.4f}')

    # Assuming `output_Q`, `imageQ`, `output_I`, `imageI`, `outputs_intensity`, and `imageA` are PyTorch tensors
    # Extracting the first channel data from the tensors
    final_output_image = output_Q.detach().cpu().numpy()[0][0]
    final_input_image = imageQ.detach().cpu().numpy()[0][0]
    final_output_image2 = output_I.detach().cpu().numpy()[0][0]
    final_input_image2 = imageI.detach().cpu().numpy()[0][0]
    final_output_image3 = outputs_intensity.detach().cpu().numpy()[0][0]
    final_input_image3 = imageA.detach().cpu().numpy()[0][0]

    # Define the output directory
    if not os.path.exists(root):
        os.makedirs(root)

    # Save images as .npy files to retain the raw data
    np.save(os.path.join(root, f'Qoutput_epoch_{epochs}.npy'), final_output_image)
    np.save(os.path.join(root, f'Qinput_epoch_{epochs}.npy'), final_input_image)
    np.save(os.path.join(root, f'Ioutput_epoch_{epochs}.npy'), final_output_image2)
    np.save(os.path.join(root, f'Iinput_epoch_{epochs}.npy'), final_input_image2)
    np.save(os.path.join(root, f'Aoutput_epoch_{epochs}.npy'), final_output_image3)
    np.save(os.path.join(root, f'Ainput_epoch_{epochs}.npy'), final_input_image3)

def calculate_entropy(image):
    """Calculate the entropy of a batch of images"""
    batch_size, _, _, _ = image.size()
    hist = torch.histc(image, bins=256, min=0, max=1)
    hist = hist / torch.sum(hist)  # Normalize histogram
    hist = hist + 1e-6  # To avoid log(0)
    entropy = -torch.sum(hist * torch.log(hist))
    return entropy / batch_size