import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- 1. DATALOADER SETUP ---
class MultimodalPDDataset(Dataset):
    def __init__(self, csv_file):
        print(f"Loading data from {csv_file}...")
        self.data = pd.read_csv(csv_file)
        
        # Ground Truth Labels
        self.labels = torch.tensor(self.data['Label_Binary'].values, dtype=torch.float32).unsqueeze(1)
        
        # Define the exact columns you currently have in your CSV
        audio_cols = ['F0_Hz', 'Jitter_Pct', 'Shimmer_Pct', 'HNR_dB']
        visual_cols = ['MAR_Mean', 'MAR_Variance']
        
        # Fill any missing values (NaN) with 0 to prevent PyTorch from crashing
        self.data[audio_cols] = self.data[audio_cols].fillna(0)
        self.data[visual_cols] = self.data[visual_cols].fillna(0)
        
        # Normalize the data so large numbers don't overpower small decimals
        self.scaler_audio = StandardScaler()
        self.scaler_visual = StandardScaler()
        
        self.audio_features = torch.tensor(self.scaler_audio.fit_transform(self.data[audio_cols]), dtype=torch.float32)
        self.visual_features = torch.tensor(self.scaler_visual.fit_transform(self.data[visual_cols]), dtype=torch.float32)

        self.num_audio_feats = len(audio_cols)
        self.num_visual_feats = len(visual_cols)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.audio_features[idx], self.visual_features[idx], self.labels[idx]

# --- 2. THE LATE-FUSION ARCHITECTURE ---
class ParkinsonLateFusionNet(nn.Module):
    def __init__(self, num_audio, num_visual):
        super(ParkinsonLateFusionNet, self).__init__()
        
        # Audio Expert Branch
        self.audio_branch = nn.Sequential(
            nn.Linear(num_audio, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
        
        # Visual/Kinematic Expert Branch
        self.visual_branch = nn.Sequential(
            nn.Linear(num_visual, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
        
        # The Fusion Judge
        self.fusion_layer = nn.Sequential(
            nn.Linear(4 + 4, 8), # 4 from audio + 4 from visual
            nn.ReLU(),
            nn.Linear(8, 1),     # 1 output (PD Probability)
            nn.Sigmoid()         # Squeezes output between 0.0 and 1.0
        )

    def forward(self, audio_x, visual_x):
        a_out = self.audio_branch(audio_x)
        v_out = self.visual_branch(visual_x)
        fused = torch.cat((a_out, v_out), dim=1) # The actual Late Fusion step
        out = self.fusion_layer(fused)
        return out

# --- 3. THE TRAINING LOOP ---
def train():
    # Load the dataset
    dataset = MultimodalPDDataset('master_training_data.csv')
    
    # Batch size is small because we only have 10 rows right now
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Initialize the Model
    model = ParkinsonLateFusionNet(dataset.num_audio_feats, dataset.num_visual_feats)
    
    # Define how we calculate error (Binary Cross Entropy for Yes/No classification)
    criterion = nn.BCELoss() 
    
    # Define the Optimizer (Adam is standard, lr is learning rate)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 50 # Number of times the AI will look at the whole dataset
    
    print("\n🚀 Beginning Training Phase...\n")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for audio_batch, visual_batch, label_batch in dataloader:
            # 1. Zero out the gradients from the last step
            optimizer.zero_grad()
            
            # 2. Forward Pass: Ask the model to guess
            predictions = model(audio_batch, visual_batch)
            
            # 3. Calculate the Error (Loss)
            loss = criterion(predictions, label_batch)
            
            # 4. Backward Pass: Calculate how to fix the error
            loss.backward()
            
            # 5. Optimize: Adjust the weights
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Track Accuracy
            predicted_classes = (predictions > 0.5).float()
            correct_predictions += (predicted_classes == label_batch).sum().item()
            total_samples += label_batch.size(0)
            
        # Calculate epoch statistics
        avg_loss = epoch_loss / len(dataloader)
        accuracy = (correct_predictions / total_samples) * 100
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    print("\n✅ Training Complete. The model successfully mapped audio/visual features to clinical labels.")
    
    # Save the trained model weights
    torch.save(model.state_dict(), "parkinsons_late_fusion_weights.pth")
    print("💾 Model weights saved to 'parkinsons_late_fusion_weights.pth'")

if __name__ == "__main__":
    train()