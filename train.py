import datetime
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, TotalVariation
from model import N2V_Unet, UNetSharedEncoder
from dataloader import img_loader, img_loader_FLIM
from tqdm import tqdm
import tifffile as tiff

def train_model(epochs, batch_size, lr, root, noise_levels, types):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = img_loader(root, batch_size, noise_levels, types)
    eval_loader = img_loader(root, batch_size, noise_levels, types, train=False)

    model = N2V_Unet(in_channels=1, out_channels=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f'runs/DNFLIM_experiment_{timestamp}')

    psnr_metric = PeakSignalNoiseRatio().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure().to(device)

    best_eval_loss = float('inf')
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training
    patience = 10  # for early stopping
    trigger = 0
    alpha = 0.5  # weight for the Noise2Void loss

    for epoch in range(epochs):
        model.train()
        running_loss, running_psnr, running_ssim = 0.0, 0.0, 0.0
        num_batches = 0

        for inputs, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                # Noise2Void masked loss
                mask = (torch.rand_like(inputs) > 0.95).float()
                masked_input = inputs * (1 - mask)
                masked_output = outputs * mask
                supervised_loss = criterion(outputs, labels)
                n2v_loss = criterion(masked_output, masked_input)
                loss = alpha * n2v_loss + (1 - alpha) * supervised_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
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
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.4f}, PSNR: {avg_epoch_psnr:.4f}, SSIM: {avg_epoch_ssim:.4f}')

        # Evaluation Loop
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

        # Learning rate scheduling based on evaluation loss
        scheduler.step(avg_eval_loss)

        # Checkpointing and Early Stopping
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            torch.save(model.state_dict(), './model2_mix/best_dnflim.pth')
            print(f'Saved Best Model at Epoch {epoch + 1} with Eval Loss: {avg_eval_loss:.4f}')
            trigger = 0
        else:
            trigger += 1
            if trigger >= patience:
                print("Early stopping triggered")
                break

    torch.save(model.state_dict(), './model2_mix/dnflim.pth')
    writer.close()

def train_FLIM_model(config):
    """
    Train the UNetSharedEncoder model on FLIM data using a mix of losses:
      - Content loss (MSE between reconstructed and input images)
      - SSIM loss (to enforce structural similarity with the intensity output)
      - MSE loss between the fused output and the intensity output
      - Total Variation (TV) loss for smoothness
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loader (assumed to be implemented elsewhere)
    train_loader = img_loader_FLIM(config.root, config.batch_size, config.num_augmentations, config.types)
    
    # Initialize models
    model_FLIM = UNetSharedEncoder(in_channels=1, base_channels=config.base_channels, out_channels=1).to(device)
    model_intensity = N2V_Unet(in_channels=1, out_channels=1).to(device)
    model_intensity.load_state_dict(torch.load(config.intensity_model_path))
    model_intensity.eval()
    
    optimizer = optim.Adam(model_FLIM.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    criterion = nn.MSELoss()
    tv_loss = TotalVariation().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    global_step = 0
    
    for epoch in range(config.epochs):
        model_FLIM.train()
        running_loss = 0.0
        epoch_start = time.time()
        
        for batch_idx, (imageI, imageQ, imageA) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")):
            imageI, imageQ, imageA = imageI.to(device), imageQ.to(device), imageA.to(device)
            optimizer.zero_grad()
            
            with torch.no_grad():
                outputs_intensity = model_intensity(imageA)
            
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                out_Q, out_I, out_A = model_FLIM(imageQ, imageI)
                ssim_loss = (2 - ssim_metric(out_Q, outputs_intensity) - ssim_metric(out_I, outputs_intensity))
                mse_loss = criterion(out_A, outputs_intensity)
                content_loss = criterion(out_Q, imageQ) + criterion(out_I, imageI)
                tv_loss_value = tv_loss(out_I) + tv_loss(out_Q)
                loss = (config.alpha * content_loss +
                        config.alpha * config.ssim_weight * ssim_loss +
                        config.alpha * config.tv_weight * tv_loss_value +
                        mse_loss)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            global_step += 1
        
        avg_loss = running_loss / len(train_loader)
        scheduler.step(avg_loss)
        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch+1}/{config.epochs}], Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
        
        if (epoch + 1) % config.checkpoint_interval == 0:
            save_images(config.root +"/"+config.types[0], out_Q, imageQ, out_I, imageI, out_A, imageA, outputs_intensity)
    

def save_images(root, output_Q, imageQ, output_I, imageI, output_A, imageA, outputs_intensity):
    """
    Save denoised FLIM images directly as TIFF files, along with lifetime calculation.
    """
    # Convert tensors to numpy
    final_output_Q = output_Q.detach().cpu().numpy()[0][0]
    final_output_I = output_I.detach().cpu().numpy()[0][0]
    final_output_A = output_A.detach().cpu().numpy()[0][0]

    input_image_A = imageA.detach().cpu().numpy()[0][0]
    input_image_I = imageI.detach().cpu().numpy()[0][0]
    input_image_Q = imageQ.detach().cpu().numpy()[0][0]

    # Create output directory
    output_dir = os.path.join(root)
    os.makedirs(output_dir, exist_ok=True)
    
    # Avoid division by zero
    epsilon = 1e-11

    # Compute G and S for outputs
    G_output = final_output_I / (final_output_A + epsilon)
    S_output = final_output_Q / (final_output_A + epsilon)
    image_data_A_output = outputs_intensity.detach().cpu().numpy()[0][0]

    
    os.makedirs(output_dir, exist_ok=True)

    # Save outputs as .tif images
    tiff.imwrite(os.path.join(output_dir, 'imageG.tif'), G_output.astype(np.float32))
    tiff.imwrite(os.path.join(output_dir, 'imageS.tif'), S_output.astype(np.float32))
    tiff.imwrite(os.path.join(output_dir, 'imageI.tif'), image_data_A_output.astype(np.float32))

    print(f"Saved outputs for {output_dir}")
    