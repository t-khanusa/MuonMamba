#!/usr/bin/env python3
"""
Example usage of MuonMamba: Mamba with Momentum + Newton-Schulz5 Orthogonalization

This script demonstrates how to:
1. Create a MuonMamba model
2. Run forward pass (training)
3. Run autoregressive generation (inference)
"""

import torch
from mamba_ssm.modules.muon_mamba import MuonMamba, MuonMambaConfig


def example_training():
    """Example: Training with MuonMamba"""
    print("="*80)
    print("EXAMPLE 1: Training with MuonMamba")
    print("="*80)
    
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
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Beta (momentum): {config.momentum_beta}")
    print(f"Alpha (scaling): {config.momentum_alpha}")
    print(f"Newton-Schulz5: {'ENABLED' if config.momentum_beta > 0 else 'DISABLED'}")
    print()
    
    # Create sample input
    batch_size = 2
    seq_len = 256  # Reduced for stability
    x = torch.randn(batch_size, seq_len, config.d_model).cuda()
    
    # Forward pass
    print(f"Input shape: {x.shape}")
    # with torch.cuda.amp.autocast():  # Mixed precision
    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print()
    
    # Backward pass (for training)
    loss = output.mean()
    loss.backward()
    print(f"Loss: {loss.item():.6f}")
    print("✓ Backward pass successful")
    print()


def example_generation():
    """Example: Autoregressive generation with MuonMamba"""
    print("="*80)
    print("EXAMPLE 2: Autoregressive Generation")
    print("="*80)
    
    # Create smaller config for faster generation
    config = MuonMambaConfig(
        d_model=128,
        n_layers=2,
        d_state=64,
        momentum_beta=0.9,
        momentum_alpha=1.0,
    )
    
    model = MuonMamba(config).cuda().eval()
    print(f"Model: {config.n_layers} layers, d_model={config.d_model}")
    print()
    
    # Allocate KV cache for each layer
    batch_size = 16
    caches = []
    for layer in model.layers:
        cache = layer.mixer.mamba.allocate_inference_cache(batch_size, max_seqlen=0)
        caches.append(cache)
    print(f"Cache allocated for batch_size={batch_size}")
    
    # Generate tokens autoregressively
    num_tokens_to_generate = 10
    current_token = torch.randn(batch_size, config.d_model).cuda()  # (B, D)
    
    generated = [current_token.unsqueeze(1)]  # Store as (B, 1, D) for concatenation
    print(f"\nGenerating {num_tokens_to_generate} tokens...")
    
    with torch.no_grad():
        for i in range(num_tokens_to_generate):
            # Single-step forward with caching (expects B, D)
            output, caches = model.step(current_token, caches)
            generated.append(output.unsqueeze(1))  # Store as (B, 1, D)
            current_token = output  # (B, D)
            
            if (i + 1) % 5 == 0:
                print(f"  Generated {i + 1} tokens")
    
    # Concatenate all generated tokens
    generated_sequence = torch.cat(generated, dim=1)  # (B, L+num_gen, D)
    print(f"\nFinal sequence shape: {generated_sequence.shape}")
    print("✓ Generation successful")
    print()


def example_compare_momentum_values():
    """Example: Compare different momentum settings"""
    print("="*80)
    print("EXAMPLE 3: Comparing Momentum Settings")
    print("="*80)
    
    configs = [
        ("No momentum (standard Mamba)", 0.0, 1.0),
        ("Low momentum", 0.5, 1.0),
        ("Medium momentum", 0.8, 1.0),
        ("High momentum", 0.9, 1.0),
    ]
    
    batch_size = 2
    seq_len = 512
    d_model = 64
    
    x = torch.randn(batch_size, seq_len, d_model).cuda()
    
    print(f"Input: {x.shape}, norm={x.norm().item():.4f}")
    print()
    
    for name, beta, alpha in configs:
        config = MuonMambaConfig(
            d_model=d_model,
            n_layers=2,
            d_state=64,
            momentum_beta=beta,
            momentum_alpha=alpha,
        )
        
        model = MuonMamba(config).cuda().eval()
        
        with torch.no_grad():
            output = model(x)
        
        ns5_status = "✓ NS5 enabled" if beta > 0 else "✗ NS5 disabled"
        print(f"{name:30s} (β={beta:.1f}, α={alpha:.1f})")
        print(f"  Output norm: {output.norm().item():.4f}")
        print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print(f"  {ns5_status}")
        print()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("MuonMamba Examples")
    print("Mamba with Momentum + Newton-Schulz5 Orthogonalization")
    print("="*80 + "\n")
    
    # Run examples
    example_training()
    example_generation()
    example_compare_momentum_values()
    
    print("="*80)
    print("All examples completed successfully!")
    print("="*80)

