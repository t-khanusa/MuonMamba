import torch
from mamba_ssm.modules.muon_mamba import MuonMamba, MuonMambaConfig
from calflops import calculate_flops
from mamba_ssm.modules.mamba_simple import Mamba
    
# Create configuration
config = MuonMambaConfig(
    d_model=128,           # Model dimension (reduced for stability)
    n_layers=4,            # Number of layers
    d_state=16,            # SSM state dimension
    expand_factor=2,       # Expansion factor (d_inner = expand * d_model)
    
    # MuonMamba parameters
    momentum_beta=0.9,     # β - momentum decay (0.5-0.9 recommended)
    momentum_alpha=1.0,    # α - momentum scaling (0.5-1.5 recommended)
    
    # When beta > 0, Newton-Schulz5 is automatically applied!
)

# Create model
model = MuonMamba(config).cuda()
model = Mamba(d_model=128, d_state=64, beta=0.9, alpha=1.0).cuda()
print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
print(f"Beta (momentum): {config.momentum_beta}")
print(f"Alpha (scaling): {config.momentum_alpha}")
print(f"Newton-Schulz5: {'ENABLED' if config.momentum_beta > 0 else 'DISABLED'}")
print()

# Create sample input
batch_size = 2
seq_len = 512  # Reduced for stability
x = torch.randn(batch_size, seq_len, config.d_model).cuda()

# Forward pass
print(f"Input shape: {x.shape}")

output = model(x)
flops = calculate_flops(model, input_shape=(batch_size, seq_len, config.d_model))
print(f"FLOPS: {flops}")
print(f"Output shape: {output.shape}")
print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
print()

# Backward pass (for training)
loss = output.mean()
loss.backward()
print(f"Loss: {loss.item():.6f}")
print("✓ Backward pass successful")
print()