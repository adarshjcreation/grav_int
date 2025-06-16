import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
from torch.autograd import Variable
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    """2D Spectral Convolution Layer"""
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        # Complex weights for Fourier coefficients
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, 
                                                           self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, 
                                                           self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        """Complex multiplication for 2D tensors"""
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), 
                           x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    """2D Fourier Neural Operator with improved architecture"""
    def __init__(self, modes1, modes2, width, input_dim=1, output_dim=1, n_layers=4):
        super(FNO2d, self).__init__()
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        
        # Input projection with proper initialization
        self.fc0 = nn.Linear(input_dim + 2, self.width)
        
        # Fourier layers
        self.conv_layers = nn.ModuleList([
            SpectralConv2d(self.width, self.width, self.modes1, self.modes2) 
            for _ in range(self.n_layers)
        ])
        
        # Local convolution layers (W in the paper) - with proper kernel size
        self.w_layers = nn.ModuleList([
            nn.Conv2d(self.width, self.width, 1) 
            for _ in range(self.n_layers)
        ])
        
        # Batch normalization for stability
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(self.width) 
            for _ in range(self.n_layers)
        ])
        
        # Output projection with more capacity for complex mappings
        self.fc1 = nn.Linear(self.width, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.activation = F.gelu
        
        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        """Proper weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_grid(self, shape, device):
        """Generate coordinate grid"""
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

    def forward(self, x):
        # Input format: (batch, channels, height, width)
        # Convert to (batch, height, width, channels) for processing
        x = x.permute(0, 2, 3, 1)
        
        # Get coordinate grid
        batch_size, height, width, channels = x.shape
        grid = self.get_grid((batch_size, height, width), x.device)
        
        # Concatenate input with grid coordinates
        x = torch.cat((x, grid), dim=-1)
        
        # Input projection
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # Convert back to (batch, channels, height, width)
        
        # Store residual connection from input projection
        x_input = x
        
        # Fourier layers with residual connections and normalization
        for i in range(self.n_layers):
            x_residual = x
            
            x1 = self.conv_layers[i](x)
            x2 = self.w_layers[i](x)
            x = x1 + x2
            
            # Add residual connection
            x = x + x_residual
            
            # Apply batch normalization
            x = self.bn_layers[i](x)
            
            if i < self.n_layers - 1:
                x = self.activation(x)
        
        # Global residual connection
        x = x + x_input
        
        # Output projection
        x = x.permute(0, 2, 3, 1)  # (batch, height, width, channels)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        # Convert output back to (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2)
        
        return x
