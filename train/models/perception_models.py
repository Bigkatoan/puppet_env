import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- Edge-Autoencoder for Visual Feature Extraction (Section 4.2.1) ---

class EdgeAutoencoder(nn.Module):
    """
    A CNN-based Autoencoder with a hybrid MSE and Sobel edge loss.
    The architecture is based on Tables 5 and 6 of the documentation.
    """
    def __init__(self, latent_dim=128): # Latent dimension is 128 [cite: 386]
        super().__init__()
        
        # --- Encoder Architecture (Table 5) [cite: 350, 351] ---
        self.encoder = nn.Sequential(
            # Input shape: (N, 3, 256, 256)
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2), # -> (N, 32, 128, 128)
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2), # -> (N, 64, 64, 64)
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2), # -> (N, 128, 32, 32)
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2), # -> (N, 256, 16, 16)

            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2), # -> (N, 256, 8, 8)
            
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, latent_dim) # Bottleneck
        )
        
        # --- Decoder Architecture (Table 6) [cite: 353, 354] ---
        self.decoder_fc = nn.Linear(latent_dim, 256 * 8 * 8)
        
        self.decoder = nn.Sequential(
            # Reshape to (N, 256, 8, 8) happens after the fc layer
            nn.Upsample(scale_factor=2, mode='nearest'), # -> (N, 256, 16, 16)
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode='nearest'), # -> (N, 256, 32, 32)
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode='nearest'), # -> (N, 128, 64, 64)
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode='nearest'), # -> (N, 64, 128, 128)
            nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode='nearest'), # -> (N, 32, 256, 256)
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(True),
            
            # Final output layer
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid() # Sigmoid to keep pixel values in [0, 1]
        )

    def forward(self, x):
        latent_vec = self.encoder(x)
        x_reshaped = self.decoder_fc(latent_vec)
        x_reshaped = x_reshaped.view(-1, 256, 8, 8)
        reconstructed_x = self.decoder(x_reshaped)
        return reconstructed_x, latent_vec

def sobel_loss(reconstruction, original):
    """Calculates the Sobel-based edge loss[cite: 346]."""
    sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Move kernel to the same device as the images
    sobel_kernel_x = sobel_kernel_x.to(reconstruction.device)
    sobel_kernel_y = sobel_kernel_y.to(reconstruction.device)

    # Convert images to grayscale for edge detection
    original_gray = torch.mean(original, 1, keepdim=True)
    recon_gray = torch.mean(reconstruction, 1, keepdim=True)

    # Apply Sobel filter
    original_edges_x = F.conv2d(original_gray, sobel_kernel_x, padding=1)
    original_edges_y = F.conv2d(original_gray, sobel_kernel_y, padding=1)
    recon_edges_x = F.conv2d(recon_gray, sobel_kernel_x, padding=1)
    recon_edges_y = F.conv2d(recon_gray, sobel_kernel_y, padding=1)

    # Calculate edge magnitudes
    original_mag = torch.sqrt(original_edges_x**2 + original_edges_y**2)
    recon_mag = torch.sqrt(recon_edges_x**2 + recon_edges_y**2)

    return F.mse_loss(recon_mag, original_mag)

# --- Pose Prediction Network for Visual Localization (Section 4.2.2) ---

class PosePredictionNetwork(nn.Module):
    """
    An MLP Regressor to predict pose from a latent vector.
    The architecture is based on Table 8 of the documentation[cite: 401, 402].
    """
    def __init__(self, latent_dim=128, output_dim=3):
        super().__init__()
        self.mlp = nn.Sequential(
            # Input is the 128-dim latent vector [cite: 402]
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim) # Linear activation for regression output [cite: 402]
        )

    def forward(self, latent_vec):
        return self.mlp(latent_vec)