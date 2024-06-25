import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

def PSNR(clean_image, noisy_image, denoised_image):
    mse_noisy = torch.mean((clean_image - noisy_image) ** 2)
    mse_denoised = torch.mean((clean_image - denoised_image) ** 2)
    psnr_noisy = 20 * torch.log10(1.0 / torch.sqrt(mse_noisy))
    psnr_denoised = 20 * torch.log10(1.0 / torch.sqrt(mse_denoised))
    return psnr_noisy.item(), psnr_denoised.item()


def train(model, train_loader, val_loader, optimizer, scheduler, criterion, num_epochs=6, learning_rate=1e-3):
    train_losses = []
    val_losses = []
    train_psnrs = []
    val_psnrs = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_psnr = []
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
        for i, (noisy_images, clean_images) in progress_bar:
            noisy_images, clean_images = noisy_images.cuda(non_blocking=True), clean_images.cuda(non_blocking=True)
            optimizer.zero_grad()
            outputs = model(noisy_images)
            outputs = torch.clamp(outputs, 0, 1)
            loss = criterion(outputs, clean_images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            psnr_noisy, psnr_denoised = PSNR(clean_images, noisy_images, outputs)
            train_psnr.append(psnr_denoised)
            progress_bar.set_postfix(loss=running_loss / (i + 1), psnr=np.mean(train_psnr[-100:]))  # Average PSNR on last 100 values

            # tout les 1000 étapes on affiche une image 
            if i % 1000 == 0:
                noisy_image_np = noisy_images[0].cpu().numpy().transpose(1, 2, 0) * 255.0
                clean_image_np = clean_images[0].cpu().numpy().transpose(1, 2, 0) * 255.0
                denoised_image_np = outputs[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0

                plt.figure(figsize=(10, 3))
                plt.subplot(1, 3, 1)
                plt.imshow(clean_image_np.astype(np.uint8))
                plt.title('Original Image')
                plt.subplot(1, 3, 2)
                plt.imshow(noisy_image_np.astype(np.uint8))
                plt.title('Noisy Image')
                plt.subplot(1, 3, 3)
                plt.imshow(denoised_image_np.astype(np.uint8))
                plt.title('Denoised Image')
                plt.show()

                print(f"PSNR Noisy: {psnr_noisy}, PSNR Denoised: {psnr_denoised}")
                

        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)
        avg_train_psnr = np.mean(train_psnr[-100:])  # Average PSNR on last 100 values
        train_losses.append(avg_train_loss)
        train_psnrs.append(avg_train_psnr)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss}, Train PSNR: {avg_train_psnr}")

        model.eval()
        val_loss = 0.0
        val_psnr = []
        with torch.no_grad():
            for noisy_images, clean_images in val_loader:
                noisy_images, clean_images = noisy_images.cuda(non_blocking=True), clean_images.cuda(non_blocking=True)
                outputs = model(noisy_images)
                outputs = torch.clamp(outputs, 0, 1)
                loss = criterion(outputs, clean_images)
                val_loss += loss.item()
                psnr_noisy, psnr_denoised = PSNR(clean_images, noisy_images, outputs)
                val_psnr.append(psnr_denoised)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_psnr = np.mean(val_psnr[-100:])  # Average PSNR on last 100 values
        val_losses.append(avg_val_loss)
        val_psnrs.append(avg_val_psnr)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss}, Validation PSNR: {avg_val_psnr}")

    # Plot loss and PSNR
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Evolution')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_psnrs, label='Train PSNR')
    plt.plot(val_psnrs, label='Validation PSNR')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.title('PSNR Evolution')

    plt.tight_layout()
    plt.show()

    return train_losses, val_losses, train_psnrs, val_psnrs