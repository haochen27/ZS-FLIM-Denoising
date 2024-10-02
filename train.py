import datetime
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, TotalVariation
from model import N2V_Unet, UNet_SharedEncoder
from dataloader import img_loader, img_loader_FLIM
from tqdm import tqdm

def train_model(epochs, batch_size, lr, root, noise_levels, types):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = img_loader(root, batch_size, noise_levels, types)
    eval_loader = img_loader(root, batch_size, noise_levels, types, train=False)

    model = N2V_Unet(in_channels=1, out_channels=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f'runs/DNFLIM_experiment_{timestamp}')

    psnr_metric = PeakSignalNoiseRatio().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure().to(device)

    best_eval_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss, running_psnr, running_ssim = 0.0, 0.0, 0.0
        num_batches = 0

        for inputs, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Noise2Void Training
            mask = (torch.rand_like(inputs) > 0.95).float()
            masked_input = inputs * (1 - mask)
            masked_output = outputs * mask
            n2v_loss = criterion(masked_output, masked_input)
            loss = criterion(outputs, labels)
            
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

        model.eval()
        eval_loss, eval_psnr, eval_ssim = 0.0, 0.0, 0.0
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
        writer.add_scalar('eval_loss', avg_eval_loss, epoch)
        writer.add_scalar('eval_psnr', avg_eval_psnr, epoch)
        writer.add_scalar('eval_ssim', avg_eval_ssim, epoch)
        print(f'Epoch [{epoch + 1}/{epochs}], Eval Loss: {avg_eval_loss:.4f}, Eval PSNR: {avg_eval_psnr:.4f}, Eval SSIM: {avg_eval_ssim:.4f}')

        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            torch.save(model.state_dict(), './model/best_dnflim.pth')
            print(f'Saved Best Model at Epoch {epoch + 1} with Eval Loss: {avg_eval_loss:.4f}')

    torch.save(model.state_dict(), './model/dnflim.pth')
    writer.close()

def ZS_FLIM_train_model(epochs, batch_size, lr, root, types, num_augmentations, alpha=0.8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = img_loader_FLIM(root, batch_size, num_augmentations, types)

    model_FLIM = UNet_SharedEncoder(in_channels=1, out_channels=1).to(device)
    model_intensity = N2V_Unet(in_channels=1, out_channels=1).to(device)
    model_intensity.load_state_dict(torch.load('./model/best_dnflim.pth'))
    model_intensity.eval()
    optimizer = optim.Adam(model_FLIM.parameters(), lr=lr)
    criterion = nn.MSELoss()
    tv_loss = TotalVariation().to(device)
    psnr_metric = PeakSignalNoiseRatio().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    for epoch in range(epochs):
        if epoch % 100 == 0:
            start_time = time.time()

        model_FLIM.train()
        running_loss = 0.0
        num_batches = 0

        for imageI, imageQ, imageA in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"):
            imageI, imageQ, imageA = imageI.to(device), imageQ.to(device), imageA.to(device)
            optimizer.zero_grad()

            outputs_intensity = model_intensity(imageA)
            output_Q, output_I, output_A = model_FLIM(imageQ, imageI)
            
            ssim_loss = (2 - ssim_metric(output_Q, outputs_intensity) - ssim_metric(output_I, outputs_intensity))
            mse_loss = criterion(output_A, outputs_intensity)
            tv_loss_value = tv_loss(output_I) + tv_loss(output_Q)
            content_loss = criterion(output_Q, imageQ) + criterion(output_I, imageI)
            loss = alpha * content_loss + alpha * 1e-4 * ssim_loss + alpha * 0.5e-6 * tv_loss_value + mse_loss
            loss.backward() 
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = running_loss / num_batches
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.4f}')

        if epoch % 100 == 99:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken for 100 epochs (Epoch {epoch - 98} to {epoch + 1}): {elapsed_time:.2f} seconds")

        if epoch % 100 == 0:
            save_images(root, output_Q, imageQ, output_I, imageI, output_A, imageA, outputs_intensity)

def save_images(root, output_Q, imageQ, output_I, imageI, output_A, imageA, outputs_intensity):
    final_output_image = output_Q.detach().cpu().numpy()[0][0]
    final_input_image = imageQ.detach().cpu().numpy()[0][0]
    final_output_image2 = output_I.detach().cpu().numpy()[0][0]
    final_input_image2 = imageI.detach().cpu().numpy()[0][0]
    final_output_image3 = output_A.detach().cpu().numpy()[0][0]
    final_input_image3 = imageA.detach().cpu().numpy()[0][0]
    outputs_intensity = outputs_intensity.detach().cpu().numpy()[0][0]

    if not os.path.exists(root):
        os.makedirs(root)

    np.save(os.path.join(root, 'Qoutput.npy'), final_output_image)
    np.save(os.path.join(root, 'Qinput.npy'), final_input_image)
    np.save(os.path.join(root, 'Ioutput.npy'), final_output_image2)
    np.save(os.path.join(root, 'Iinput.npy'), final_input_image2)
    np.save(os.path.join(root, 'Aoutput.npy'), final_output_image3)
    np.save(os.path.join(root, 'Ainput.npy'), final_input_image3)
    np.save(os.path.join(root, 'intensityoutput.npy'), outputs_intensity)

    