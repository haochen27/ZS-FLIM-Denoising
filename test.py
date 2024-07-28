import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from model import N2N_Autoencoder
from dataloader import img_loader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

def test_model(batch_size, root, noise_levels, types=None, pretrained_model='./model/dnflim.pth',FLIM_flag = True):
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the test data
    test_loader = img_loader(root, batch_size, noise_levels, types, train=False)

    # Load the model and move it to the GPU if available
    model = N2N_Autoencoder(in_channels=1, out_channels=1).to(device)
    if pretrained_model:
        model.load_state_dict(torch.load(pretrained_model))
    criterion = nn.MSELoss()

    # Metrics
    psnr_metric = PeakSignalNoiseRatio().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure().to(device)

    results = []  # List to store results

    total_loss = 0.0
    total_ssim = 0.0
    total_psnr = 0.0

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing", unit="batch"):
            images, labels, file_names = data  # Assuming file_names are returned by the loader
            images, labels = images.to(device), labels.to(device)  # Move data to GPU
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Compute PSNR and SSIM
            psnr_value = psnr_metric(outputs, labels)
            ssim_value = ssim_metric(outputs, labels)
            
            total_psnr += psnr_value.item()
            total_ssim += ssim_value.item()
            
            for i in range(images.size(0)):
                results.append({
                    'File Name': file_names[i],
                    'Loss': loss.item(),
                    'SSIM': ssim_value.item(),
                    'PSNR': psnr_value.item()
                })

    # Calculate averages
    avg_loss = total_loss / len(test_loader)
    avg_ssim = total_ssim / len(test_loader)
    avg_psnr = total_psnr / len(test_loader)
    
    # Print averages
    print(f'Average loss on the test dataset: {avg_loss:.4f}')
    print(f'Average SSIM on the test dataset: {avg_ssim:.4f}')
    print(f'Average PSNR on the test dataset: {avg_psnr:.4f} dB')

    results_df = pd.DataFrame(results)
    results_df.to_csv('test_results.csv', index=False)


