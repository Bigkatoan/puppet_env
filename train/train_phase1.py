import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import os
import numpy as np
import argparse
from tqdm import tqdm

# Import the models and loss function from your perception_models.py file
from models.perception_models import EdgeAutoencoder, PosePredictionNetwork, sobel_loss

# --- Custom PyTorch Dataset for Loading Phase 1 Data ---

class PerceptionDataset(Dataset):
    """
    Custom Dataset for loading images and their corresponding poses
    from the data collected by data_collector.py.
    """
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Get image path and pose from the dataframe
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0])
        # Pose is [x, y, yaw]
        pose = self.data_frame.iloc[idx, 1:4].astype('float').values
        
        # Load image
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB

        if self.transform:
            image = self.transform(image)
            
        return image, torch.FloatTensor(pose)

# --- Image Transformation Pipeline ---

class ImageTransforms:
    """A callable class to handle image transformations."""
    def __init__(self, output_size=(256, 256)):
        # The Edge-Autoencoder expects 256x256 images as per Table 5
        self.output_size = output_size

    def __call__(self, image):
        # Resize the image
        image = cv2.resize(image, self.output_size, interpolation=cv2.INTER_AREA)
        # Normalize pixels to be between 0 and 1
        image = image.astype(np.float32) / 255.0
        # Transpose the image from (H, W, C) to (C, H, W) for PyTorch
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)

# --- Training Function for the Edge-Autoencoder ---

def train_autoencoder(args, device):
    """
    Trains the EdgeAutoencoder model as described in Section 4.2.1.
    """
    print("--- Training Edge-Autoencoder ---")
    
    # Setup dataset and dataloader
    dataset = PerceptionDataset(
        csv_file=os.path.join(args.dataset_path, 'data.csv'),
        img_dir=os.path.join(args.dataset_path, 'images'),
        transform=ImageTransforms()
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Initialize model, optimizer, and loss function
    # Latent dimension and LR from Tables 7
    model = EdgeAutoencoder(latent_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4) 
    mse_loss = nn.MSELoss()

    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        loop = tqdm(dataloader, leave=True)
        running_loss = 0.0
        for i, (images, _) in enumerate(loop):
            images = images.to(device)
            
            # Forward pass
            reconstructed_images, _ = model(images)
            
            # Calculate hybrid loss (Eq. 14)
            loss_mse = mse_loss(reconstructed_images, images)
            loss_edge = sobel_loss(reconstructed_images, images)
            total_loss = loss_mse + loss_edge
            
            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{args.epochs}]")
            loop.set_postfix(loss=total_loss.item())

    # Save the trained model
    save_path = "edge_autoencoder.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n--- Edge-Autoencoder training complete. Model saved to {save_path} ---")


# --- Training Function for the Pose Prediction Network ---

def train_pose_network(args, device):
    """
    Trains the PosePredictionNetwork model as described in Section 4.2.2.
    """
    print("--- Training Pose Prediction Network ---")

    # 1. Load the pre-trained EdgeAutoencoder to use its encoder part
    print(f"Loading pre-trained EdgeAutoencoder from: {args.ae_weights}")
    autoencoder = EdgeAutoencoder(latent_dim=128).to(device)
    autoencoder.load_state_dict(torch.load(args.ae_weights, map_location=device))
    encoder = autoencoder.encoder
    encoder.eval() # Set encoder to evaluation mode
    print("Encoder loaded successfully.")

    # 2. Setup dataset and dataloader
    dataset = PerceptionDataset(
        csv_file=os.path.join(args.dataset_path, 'data.csv'),
        img_dir=os.path.join(args.dataset_path, 'images'),
        transform=ImageTransforms()
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 3. Initialize PoseNet model, optimizer, and loss
    # Output dimension is 3 for [x, y, yaw] to match collected data
    model = PosePredictionNetwork(latent_dim=128, output_dim=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss() # As specified in Eq. 15

    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        loop = tqdm(dataloader, leave=True)
        running_loss = 0.0
        for i, (images, poses) in enumerate(loop):
            images = images.to(device)
            poses = poses.to(device)
            
            # Generate latent vectors using the frozen encoder
            with torch.no_grad():
                latent_vectors = encoder(images)
            
            # Forward pass through the PoseNet
            predicted_poses = model(latent_vectors)
            
            # Calculate loss
            loss = criterion(predicted_poses, poses)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{args.epochs}]")
            loop.set_postfix(loss=loss.item())

    # Save the trained model
    save_path = "pose_network.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n--- Pose Prediction Network training complete. Model saved to {save_path} ---")


# --- Main execution block ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Phase 1: Pre-training of Perception Models")
    
    # Arguments to select which model to train
    parser.add_argument("--model", type=str.lower, required=True, choices=["autoencoder", "posenet"], 
                        help="Which model to train: 'autoencoder' or 'posenet'.")
    
    # Common arguments
    parser.add_argument("--dataset-path", type=str, default="phase1_dataset", help="Path to the folder containing collected data.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (as per Table 7).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training (as per Table 7).")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    
    # Argument needed only when training the pose network
    parser.add_argument("--ae-weights", type=str, default="edge_autoencoder.pth", 
                        help="Path to pre-trained Edge-AE weights (required for training posenet).")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Run the selected training function
    if args.model == "autoencoder":
        train_autoencoder(args, device)
    elif args.model == "posenet":
        if not os.path.exists(args.ae_weights):
            print(f"Error: Autoencoder weights not found at '{args.ae_weights}'.")
            print("Please train the autoencoder first using --model autoencoder")
        else:
            train_pose_network(args, device)