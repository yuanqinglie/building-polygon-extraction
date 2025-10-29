
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional

# --------------------------- Tool functions ---------------------------
def gaussian_smooth(x: torch.Tensor, kernel_size: int = 3, sigma: float = 1.0) -> torch.Tensor:
    """Gaussian smoothing (fixed version): Ensure correct input dimensions + matching convolution kernel parameters"""
    # 1. Force check input dimensions (must be 4D: [B, C, H, W])
    if x.dim() != 4:
        raise ValueError(f"gaussian_smooth requires 4D tensor input, actual input dimension is {x.dim()}, shape is {x.shape}")

    device = x.device
    B, C, H, W = x.shape  # Parse dimensions: B=batch, C=channels, H=height, W=width
    center = kernel_size // 2

    # 2. Check kernel size validity
    if kernel_size % 2 == 0:
        raise ValueError(f"Gaussian kernel size must be odd, current size is {kernel_size}")

    # 3. Create grouped convolution kernel (shape [C, 1, kernel_size, kernel_size])
    # Each channel corresponds to one kernel, ensuring matching when groups=C
    kernel = torch.zeros(C, 1, kernel_size, kernel_size, device=device, dtype=torch.float32)
    for i in range(kernel_size):
        for j in range(kernel_size):
            # Calculate Gaussian function (using tensor operations to avoid type errors)
            dx = torch.tensor(i - center, device=device, dtype=torch.float32)
            dy = torch.tensor(j - center, device=device, dtype=torch.float32)
            exponent = -(dx**2 + dy**2) / (2 * sigma**2)
            kernel[:, 0, i, j] = torch.exp(exponent)  # All channels share the same kernel values

    # 4. Normalize kernel (each channel's kernel is normalized separately)
    kernel_sum = kernel.sum(dim=(2, 3), keepdim=True)  # Sum along spatial dimensions
    kernel = kernel / (kernel_sum + 1e-8)  # Avoid division by zero

    # 5. Grouped convolution (ensure parameter matching)
    try:
        return F.conv2d(
            input=x,
            weight=kernel,
            padding=center,
            groups=C  # Each channel as a group, matching kernel's output channel count C
        )
    except RuntimeError as e:
        # Capture convolution error and output detailed information
        raise RuntimeError(
            f"Gaussian smoothing convolution failed! Input shape: {x.shape}, kernel shape: {kernel.shape}, groups: {C}\n"
            f"Original error: {str(e)}"
        ) from e


def harris_response(Ix2: torch.Tensor, Iy2: torch.Tensor, Ixy: torch.Tensor, k: float = 0.04) -> torch.Tensor:
    """Calculate Harris corner response (corresponds to Harr(·) in the paper)"""
    det = Ix2 * Iy2 - Ixy**2
    trace = Ix2 + Iy2
    return det - k * trace**2
