import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
from torch.autograd import Variable
import torch.nn.functional as F


class BranchNetwork(nn.Module):
    """Branch network processes the input function u(x)"""
    def __init__(self, input_channels=1, latent_dim=128, num_sensors=100):
        super(BranchNetwork, self).__init__()
        self.num_sensors = num_sensors
        
        # CNN layers to extract features from input
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))  # Reduce spatial dimensions
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
    def forward(self, u):
        # u shape: (batch_size, 1, 32, 32)
        x = self.conv_layers(u)
        x = x.view(x.size(0), -1)  # Flatten
        branch_output = self.fc_layers(x)
        return branch_output


class TrunkNetwork(nn.Module):
    """Trunk network processes the coordinate inputs y"""
    def __init__(self, coord_dim=2, latent_dim=128, hidden_dim=128):
        super(TrunkNetwork, self).__init__()
        
        self.fc_layers = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, y):
        # y shape: (batch_size, num_points, coord_dim)
        return self.fc_layers(y)


class DeepONet(nn.Module):
    """Deep Operator Network"""
    def __init__(self, input_channels=1, output_channels=16, H=32, W=32, latent_dim=128):
        super(DeepONet, self).__init__()
        self.H = H
        self.W = W
        self.output_channels = output_channels
        self.latent_dim = latent_dim
        
        # Branch network
        self.branch_net = BranchNetwork(input_channels, latent_dim)
        
        # Trunk network
        self.trunk_net = TrunkNetwork(coord_dim=2, latent_dim=latent_dim)
        
        # Output projection for multiple channels
        self.output_projection = nn.Linear(1, output_channels)
        
        # Generate coordinate grid
        self.register_buffer('coords', self._generate_coordinates(H, W))
        
    def _generate_coordinates(self, H, W):
        """Generate normalized coordinate grid"""
        x = torch.linspace(-1, 1, W)
        y = torch.linspace(-1, 1, H)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
        return coords  # Shape: (H*W, 2)
    
    def forward(self, u):
        batch_size = u.size(0)
        num_points = self.H * self.W
        
        # Branch network output
        branch_output = self.branch_net(u)  # (batch_size, latent_dim)
        
        # Trunk network output
        coords_batch = self.coords.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, H*W, 2)
        trunk_output = self.trunk_net(coords_batch)  # (batch_size, H*W, latent_dim)
        
        # Compute dot product between branch and trunk outputs
        branch_expanded = branch_output.unsqueeze(1)  # (batch_size, 1, latent_dim)
        dot_product = torch.sum(branch_expanded * trunk_output, dim=-1)  # (batch_size, H*W)
        
        # Project to multiple output channels
        dot_product_expanded = dot_product.unsqueeze(-1)  # (batch_size, H*W, 1)
        output = self.output_projection(dot_product_expanded)  # (batch_size, H*W, output_channels)
        
        # Reshape to desired output format
        output = output.permute(0, 2, 1)  # (batch_size, output_channels, H*W)
        output = output.view(batch_size, self.output_channels, self.H, self.W)
        
        return output

  # Trainer class
  class DeepONetTrainer:
    """Training class for DeepONet"""
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
        return total_loss / len(dataloader)
    
    def train(self, train_loader, val_loader, epochs=100):
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.scheduler.step()
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}/{epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        return train_losses, val_losses
