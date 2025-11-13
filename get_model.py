
from mamba_ssm.modules.muon_mamba import MuonMamba, MuonMambaConfig
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer    
import math


class IMUTransformerEncoder(nn.Module):

    def __init__(self, window_size = 256, num_classes = 12):

        super().__init__()

        self.transformer_dim = 256

        self.input_proj = nn.Sequential(nn.Conv1d(6, self.transformer_dim, 1), nn.GELU())
                                        # nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU(),
                                        # nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU(),
                                        # nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU())

        self.window_size = window_size
        self.encode_position = True
        encoder_layer = TransformerEncoderLayer(d_model = self.transformer_dim,
                                       nhead = 8,
                                       dim_feedforward = 128,
                                       dropout = 0.1,
                                       activation = 'gelu')

        self.transformer_encoder = TransformerEncoder(encoder_layer,
                                              num_layers = 3,
                                              norm = nn.LayerNorm(self.transformer_dim))
        self.cls_token = nn.Parameter(torch.randn((1, self.transformer_dim)), requires_grad=True)

        if self.encode_position:
            self.position_embed = nn.Parameter(torch.randn(self.window_size + 1, 1, self.transformer_dim))

        self.imu_head = nn.Sequential(
            nn.LayerNorm(self.transformer_dim),
            nn.Linear(self.transformer_dim,  self.transformer_dim//4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.transformer_dim//4,  num_classes)
        )
        self.log_softmax = nn.LogSoftmax(dim=1)
        self._init_weights()


    def _init_weights(self):
        # Initialize convolutional layers
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
                
        # Specifically initialize the cls token and position embedding
        nn.init.normal_(self.cls_token, std=0.02)
        if self.encode_position:
            nn.init.normal_(self.position_embed, std=0.02)


    def forward(self, src):
        # Embed in a high dimensional space and reshape to Transformer's expected shape
        src = self.input_proj(src).permute(2, 0, 1)
        # Prepend class token
        cls_token = self.cls_token.unsqueeze(1).repeat(1, src.shape[1], 1)
        src = torch.cat([cls_token, src])
        # Add the position embedding
        if self.encode_position:
            src += self.position_embed
        # Transformer Encoder pass
        target = self.transformer_encoder(src)[0]
        # Class probability
        target = self.imu_head(target)
        return target


class MMA(nn.Module):

    def __init__(self, d_model=128, num_classes=32, input_channels=6, n_layers=2,
                 momentum_beta=0.8, momentum_alpha=1.0):
        super().__init__()
                
        # Store momentum parameters
        self.momentum_beta = momentum_beta
        self.momentum_alpha = momentum_alpha
        
        # Initial Conv1d layer: (B,6,L) -> (B,d_model,L) with BatchNorm1d and ReLU
        self.conv1d = nn.Sequential(
            nn.Conv1d(input_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        
        # Dropout after conv1d (p=0.1 as optimized)
        self.dropout1 = nn.Dropout(0.1)
        
        # Direct MambaMomentum model (much more efficient)
        config = MuonMambaConfig(
            d_model=d_model,
            n_layers=n_layers,  # Use n_layers instead of separate blocks
            momentum_beta=momentum_beta,
            momentum_alpha=momentum_alpha,
            d_state=64,
            expand_factor=2,
            d_conv=4,
        )
        
        self.mamba_momentum = MuonMamba(config)

        # Dropout before classifier (p=0.1 as optimized)
        self.dropout2 = nn.Dropout(0.1)
        
        # Final linear layer: D -> n_class
        self.classifier = nn.Linear(d_model, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Lightweight weight initialization"""
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
        elif isinstance(module, nn.Linear):
            # Special initialization for classifier
            if module is self.classifier:
                nn.init.normal_(module.weight, std=0.01)
                nn.init.constant_(module.bias, 0)
            else:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.constant_(module.weight, 1.0)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):

        # Input: (B, 6, L)
        # Conv1d + BatchNorm1d + ReLU
        x = self.conv1d(x)  # (B, 6, L) -> (B, d_model, L)
        # Dropout
        x = self.dropout1(x)        
        # Transpose for Mamba Momentum: (B, d_model, L) -> (B, L, d_model)
        x = x.transpose(1, 2)
        # Direct Mamba Momentum (much faster than separate blocks)
        x = self.mamba_momentum(x)
        # # RMS Norm
        # x = self.rms_norm(x)
        # Dropout
        x = self.dropout2(x)
        # Global Average Pooling: (B, L, d_model) -> (B, d_model)
        x = x.mean(dim=1)
        x = self.classifier(x)
       
        return x



def get_model(args, device = "cuda:0"):
    name = args.name    
    num_classes = args.num_classes
    seq_len = 512
    input_channels = args.input_channels
    n_layers = args.n_layers
    momentum_beta = args.momentum_beta
    momentum_alpha = args.momentum_alpha
    d_model = args.d_model

    if name == "MAMCA":
        from mamba_ssm.models.config_mamba import MambaConfig
        config = MambaConfig

        config.d_model = 128
        config.n_layer = 2
        # Add momentum parameters to ssm_cfg - these can be overridden when creating the model
        config.ssm_cfg = {
            "d_state": 64, 
            "d_conv": 4, 
            "expand": 2,
        }
        # CRITICAL: Set these for numerical stability
        config.residual_in_fp32 = True   # Keep residuals in FP32 for stability
        config.fused_add_norm = False    # Disable for better debugging and stability
        config.rms_norm = True           # Use RMSNorm
        
        # model = MAMCA(config = config, device = "cuda:0", length=seq_len, num_claasses=num_classes)
        # return model
    
    elif name == "IMUTransformerEncoder":
        model = IMUTransformerEncoder(window_size = seq_len, num_classes = num_classes).to("cuda")
        return model

    elif name == "MMA":
        model = MMA(d_model = d_model, num_classes = num_classes, input_channels = input_channels, n_layers = n_layers, momentum_beta = momentum_beta, momentum_alpha = momentum_alpha).to("cuda")
        return model

    else:
        raise NotImplementedError("Model {} is not implemented".format(name))


# input = torch.randn(1, 6, 512).to("cuda:0")
# model = get_model("args", device = "cuda:0")
# print(model)
# output = model(input)
# print(output.shape)