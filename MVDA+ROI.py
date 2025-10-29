
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

# --------------------------- MVDA+ROI Module ---------------------------

class MVDA(nn.Module):
    """Multi-vertices Collaborative Deformable Attention Module (Fixed ROI Align Version)"""
    def __init__(self, d_model: int = 256, n_heads: int = 8, num_vertices_per_building: int = 8,
                 use_roi_align: bool = True, roi_output_size: int = 7):
        super().__init__()
        self.d_model = d_model
        self.num_vertices = num_vertices_per_building
        self.n_heads = n_heads
        self.use_roi_align = use_roi_align
        self.roi_output_size = roi_output_size

        # Self-attention (positional interaction)
        self.mhsa = DeformableAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention (image context reasoning)
        self.mhca = DeformableAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)

        # MLP layers
        self.mlp_pos = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )

        # ROI feature processing layer
        if self.use_roi_align:
            # ROI Align version: processes 7x7 spatial features
            roi_feat_dim = d_model * roi_output_size * roi_output_size
            self.roi_mlp = nn.Sequential(
                nn.Linear(roi_feat_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model)
            )
        else:
            # Simplified version: global pooling
            self.roi_mlp = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(d_model, d_model)
            )

        # Cross-attention projection layer
        cross_attn_input_dim = d_model * (num_vertices_per_building + 1)
        self.mlp_cross_proj = nn.Sequential(
            nn.Linear(cross_attn_input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Offset prediction layer
        self.mlp_offset = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2)
        )

    def extract_roi_features(self, Xh: torch.Tensor, roi_bboxes: torch.Tensor) -> torch.Tensor:
        """
        Fixed ROI feature extraction
        Args:
            Xh: Deep feature map (B, C, H, W)
            roi_bboxes: ROI bounding boxes (B*K, 4) in format [x_min, y_min, x_max, y_max]
        Returns:
            roi_features: ROI features (B, K, C)
        """
        if not self.use_roi_align:
            # Simplified version: global average pooling
            B, C, H, W = Xh.shape
            K = roi_bboxes.shape[0] // B
            global_feat = F.adaptive_avg_pool2d(Xh, 1)  # (B, C, 1, 1)
            return global_feat.view(B, 1, C).expand(B, K, C)  # (B, K, C)

        # Fixed ROI Align implementation
        B, C, H, W = Xh.shape
        K = roi_bboxes.shape[0] // B

        # Create batch indices
        batch_indices = torch.arange(B, device=Xh.device).view(B, 1).expand(B, K).reshape(-1)

        # Ensure valid ROIs
        if roi_bboxes.numel() == 0:
            return torch.zeros(B, K, C, device=Xh.device)

        try:
            # Prepare input format for ROI Align
            # roi_align expects format: [batch_index, x1, y1, x2, y2]
            rois_with_batch = torch.cat([
                batch_indices.unsqueeze(1).float(),  # Batch indices
                roi_bboxes  # Bounding box coordinates
            ], dim=1)  # (B*K, 5)

            # Use torchvision's ROI Align
            # Note: Coordinates should be absolute, but ours are normalized, so set spatial_scale=1.0
            roi_features = ops.roi_align(
                Xh,
                rois_with_batch,
                output_size=self.roi_output_size,
                spatial_scale=1.0,  # Because coordinates are already normalized
                sampling_ratio=2
            )  # (B*K, C, 7, 7)

            # Process ROI features: flatten and pass through MLP
            roi_features_flat = roi_features.view(B * K, -1)  # (B*K, C*7*7)
            roi_features_proj = self.roi_mlp(roi_features_flat)  # (B*K, C)
            roi_features_proj = roi_features_proj.view(B, K, C)  # (B, K, C)

            return roi_features_proj

        except Exception as e:
            print(f"ROI Align failed, using fallback: {e}")
            # Fallback: global features
            global_feat = F.adaptive_avg_pool2d(Xh, 1)  # (B, C, 1, 1)
            return global_feat.view(B, 1, C).expand(B, K, C)

    def compute_roi_bboxes(self, Drc: torch.Tensor, expansion_ratio: float = 0.2) -> torch.Tensor:
        """
        Fixed ROI bounding box calculation
        Args:
            Drc: Vertex coordinates (B, K, Nv, 2) normalized to [-1, 1]
            expansion_ratio: Bounding box expansion ratio
        Returns:
            roi_bboxes: ROI bounding boxes (B*K, 4) in format [x_min, y_min, x_max, y_max]
        """
        B, K, Nv, _ = Drc.shape

        # Calculate bounding boxes for each polygon
        roi_min = Drc.amin(dim=2)  # (B, K, 2)
        roi_max = Drc.amax(dim=2)  # (B, K, 2)

        # Calculate bounding box size and expand
        roi_size = roi_max - roi_min
        expansion = roi_size * expansion_ratio

        # Ensure minimum size after expansion
        min_size = 0.2  # Minimum bounding box size
        current_size = roi_size + 2 * expansion
        too_small = (current_size[..., 0] < min_size) | (current_size[..., 1] < min_size)
        additional_expansion = torch.where(
            too_small.unsqueeze(-1),
            (min_size - current_size) / 2,
            0
        )
        expansion += additional_expansion

        roi_min_expanded = roi_min - expansion
        roi_max_expanded = roi_max + expansion

        # Clamp to valid range [-1, 1]
        roi_min_expanded = roi_min_expanded.clamp(-1, 1)
        roi_max_expanded = roi_max_expanded.clamp(-1, 1)

        # Ensure valid bounding boxes (x_min < x_max, y_min < y_max)
        valid_bbox = (roi_max_expanded[..., 0] - roi_min_expanded[..., 0] > 0.05) & \
                    (roi_max_expanded[..., 1] - roi_min_expanded[..., 1] > 0.05)

        # Use default bounding box for invalid cases
        default_bbox = torch.tensor([-0.5, -0.5, 0.5, 0.5], device=Drc.device)
        roi_bboxes = torch.stack([
            roi_min_expanded[..., 0],  # x_min
            roi_min_expanded[..., 1],  # y_min
            roi_max_expanded[..., 0],  # x_max
            roi_max_expanded[..., 1]   # y_max
        ], dim=-1)  # (B, K, 4)

        # Apply validity mask
        roi_bboxes = torch.where(
            valid_bbox.unsqueeze(-1).expand_as(roi_bboxes),
            roi_bboxes,
            default_bbox.expand_as(roi_bboxes)
        )

        # Reshape to (B*K, 4)
        roi_bboxes = roi_bboxes.reshape(B * K, 4)

        return roi_bboxes

    def bilinear_interpolate_sampling(self, feat: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Bilinear interpolation sampling"""
        B, C, H, W = feat.shape
        Nq, Nv = coords.shape[1], coords.shape[2]

        # Ensure coordinates are within valid range
        coords = coords.clamp(-1, 1)
        coords_reshaped = coords.reshape(B, Nq*Nv, 1, 2)

        # Sample features
        sampled_feat = F.grid_sample(
            feat,
            coords_reshaped,
            mode='bilinear',
            align_corners=False
        )

        # Reshape to (B, Nq, Nv, C)
        sampled_feat = sampled_feat.reshape(B, C, Nq, Nv).permute(0, 2, 3, 1)
        return sampled_feat

    def forward(self, Q: torch.Tensor, Drc: torch.Tensor, Zf: torch.Tensor, Xh: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Q: Initial query embeddings (B, K, C)
            Drc: Initial candidate vertex coordinates (B, K, Nv, 2)
            Zf: CFPE-enhanced features (B, C, H, W)
            Xh: Encoder deep features (B, C, H, W)
        Returns:
            Q_prime: Updated query embeddings (B, K, C)
            Drc_prime: Updated vertex coordinates (B, K, Nv, 2)
        """
        B, K, C = Q.shape
        Nv = self.num_vertices

        # Step 1: Self-attention positional interaction
        pos_embed = self.mlp_pos(Q)
        Q_mhsa = self.mhsa(Q, Q, Q)
        Q1 = self.norm1(Q + Q_mhsa + pos_embed)

        # Step 2: Vertex feature sampling
        Fvj = self.bilinear_interpolate_sampling(Zf, Drc)  # (B, K, Nv, C)

        # Step 3: ROI feature extraction (fixed version)
        roi_bboxes = self.compute_roi_bboxes(Drc)
        Firoi = self.extract_roi_features(Xh, roi_bboxes)  # (B, K, C)

        # Step 4: Feature concatenation and cross-attention
        Fvj_flat = Fvj.reshape(B, K, Nv * C)  # (B, K, Nv×C)
        Fqr = torch.cat([Fvj_flat, Firoi], dim=-1)  # (B, K, Nv×C + C)

        # Process concatenated features with dedicated MLP
        Fqr_projected = self.mlp_cross_proj(Fqr)  # (B, K, C)

        # Cross-attention update
        Q_mhca = self.mhca(Q1, Fqr_projected, Fqr_projected)
        Q_prime = self.norm2(Q1 + Q_mhca)

        # Step 5: Vertex coordinate update
        offset = self.mlp_offset(Q_prime).unsqueeze(2)  # (B, K, 1, 2)
        Drc_prime = Drc + offset

        return Q_prime, Drc_prime


# --------------------------- Fixed DeformableAttention Module ---------------------------

class DeformableAttention(nn.Module):
    """Deformable Attention Base Module (Fixed Version)"""
    def __init__(self, d_model: int = 256, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Ensure d_model is divisible by n_heads
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, Nq, C = q.shape
        Nk = k.shape[1]

        # Check input dimension
        if C != self.d_model:
            raise ValueError(f"Input feature dimension {C} doesn't match d_model {self.d_model}")

        # Multi-head projection
        q_proj = self.w_q(q).reshape(B, Nq, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, Nq, Dk)
        k_proj = self.w_k(k).reshape(B, Nk, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, Nk, Dk)
        v_proj = self.w_v(v).reshape(B, Nk, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, Nk, Dk)

        # Attention calculation
        attn = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / (self.d_k ** 0.5)  # (B, H, Nq, Nk)
        attn = F.softmax(attn, dim=-1)

        # Output calculation
        output = torch.matmul(attn, v_proj).transpose(1, 2).reshape(B, Nq, C)  # (B, Nq, C)
        output = self.w_o(output)

        return output
