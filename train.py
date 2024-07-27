import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from model import N2N_Autoencoder
from dataloader import img_loader
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from model import N2N_Autoencoder
from dataloader import img_loader
from tqdm import tqdm

def train_model(epochs, batch_size, lr, root, noise_levels, types):
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the training and evaluation data
    train_loader = img_loader(root, batch_size, noise_levels, types)
    eval_loader = img_loader(root, batch_size, noise_levels, types, train=False)

    # Initialize the model, loss function, and optimizer
    model = N2N_Autoencoder(in_channels=1, out_channels=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize TensorBoard writer
    writer = SummaryWriter('runs/DNFLIM_experiment')

    # Initialize metrics
    psnr_metric = PeakSignalNoiseRatio().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure().to(device)

    # Training loop
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_psnr = 0.0
        running_ssim = 0.0
        num_batches = 0

        for i, (inputs, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")):
            # Move inputs to GPU
            inputs = inputs.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, inputs)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate running loss and metrics
            running_loss += loss.item()
            running_psnr += psnr_metric(outputs, inputs).item()
            running_ssim += ssim_metric(outputs, inputs).item()
            num_batches += 1

            # Log every 100 batches
            if (i + 1) % 100 == 0:
                avg_loss = running_loss / 100
                avg_psnr = running_psnr / 100
                avg_ssim = running_ssim / 100
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}')
                writer.add_scalar('training loss', avg_loss, epoch * len(train_loader) + i)
                writer.add_scalar('training psnr', avg_psnr, epoch * len(train_loader) + i)
                writer.add_scalar('training ssim', avg_ssim, epoch * len(train_loader) + i)
                running_loss = 0.0
                running_psnr = 0.0
                running_ssim = 0.0

        # End of epoch training metrics
        avg_epoch_loss = running_loss / num_batches
        avg_epoch_psnr = running_psnr / num_batches
        avg_epoch_ssim = running_ssim / num_batches
        writer.add_scalar('epoch_loss', avg_epoch_loss, epoch)
        writer.add_scalar('epoch_psnr', avg_epoch_psnr, epoch)
        writer.add_scalar('epoch_ssim', avg_epoch_ssim, epoch)

        # Evaluation
        model.eval()  # Set model to evaluation mode
        eval_loss = 0.0
        eval_psnr = 0.0
        eval_ssim = 0.0
        eval_batches = 0

        with torch.no_grad():
            for eval_inputs, _ in eval_loader:
                eval_inputs = eval_inputs.to(device)
                eval_outputs = model(eval_inputs)
                eval_loss += criterion(eval_outputs, eval_inputs).item()
                eval_psnr += psnr_metric(eval_outputs, eval_inputs).item()
                eval_ssim += ssim_metric(eval_outputs, eval_inputs).item()
                eval_batches += 1

        avg_eval_loss = eval_loss / eval_batches
        avg_eval_psnr = eval_psnr / eval_batches
        avg_eval_ssim = eval_ssim / eval_batches
        print(f'Epoch [{epoch + 1}/{epochs}], Eval Loss: {avg_eval_loss:.4f}, Eval PSNR: {avg_eval_psnr:.4f}, Eval SSIM: {avg_eval_ssim:.4f}')
        writer.add_scalar('eval loss', avg_eval_loss, epoch)
        writer.add_scalar('eval psnr', avg_eval_psnr, epoch)
        writer.add_scalar('eval ssim', avg_eval_ssim, epoch)

    # Save the final model state
    torch.save(model.state_dict(), 'dnflim.pth')
    writer.close()




