
# --------------------------- 2.3 Dynamic Query Optimizer (DQO) ---------------------------
class DQO(nn.Module):
    """Dynamic Query Optimizer module (fully fixed version)"""
    def __init__(self, in_channels: int = 256, num_queries: int = 100, num_vertices_per_building: int = 8):
        super().__init__()
        self.num_queries = num_queries
        self.num_vertices_per_building = num_vertices_per_building
        self.in_channels = in_channels

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, in_channels, 64, 64))

        # Candidate vertex coordinate prediction branch
        self.conv_fc = nn.Conv2d(in_channels, num_vertices_per_building * 2, kernel_size=3, stride=2, padding=1)

        # Instance score prediction branch - fix: use correct structure
        self.score_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, C, 1, 1)
            nn.Flatten(),             # (B, C)
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 2)          # (B, 2)
        )

        # Query embedding projection - fix: ensure correct input-output dimensions
        self.query_proj = nn.Linear(in_channels, in_channels)

    def forward(self, Zf: torch.Tensor, Xh: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = Zf.shape

        # Step 1: Add positional encoding
        pos_embed_resized = F.interpolate(self.pos_embed, size=(H, W), mode='bilinear', align_corners=False)
        Zf_with_pos = Zf + pos_embed_resized

        # Step 2: Predict candidate vertex coordinates
        Fc = self.conv_fc(Zf_with_pos)  # (B, Nv×2, H/2, W/2)
        S_h, S_w = Fc.shape[2], Fc.shape[3]
        S = S_h * S_w

        # Step 3: Predict instance scores - fix: use Xh instead of Zf
        Fs_logits = self.score_conv(Xh)  # (B, 2)
        Fs = F.softmax(Fs_logits, dim=-1)

        # Step 4: Generate spatial score map
        # Expand instance scores to spatial dimensions
        spatial_scores = Fs[:, 1:2]  # (B, 1) - building class score
        spatial_scores = spatial_scores.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)
        spatial_scores = spatial_scores.expand(B, 1, S_h, S_w)  # (B, 1, S_h, S_w)

        # Step 5: Select Top-K query positions
        spatial_scores_flat = spatial_scores.view(B, -1)  # (B, S)
        topk_scores, topk_indices = torch.topk(spatial_scores_flat, k=self.num_queries, dim=1)  # (B, K)

        # Step 6: Extract query embeddings - fix: extract from Zf_with_pos instead of Fc
        Zf_flat = Zf_with_pos.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, H*W, C)

        # Since grid sizes of Zf_flat and topk_indices don't match, we need adjustment
        # Simplified method: generate query embeddings from positions corresponding to Fc
        Q_topk = torch.zeros(B, self.num_queries, C, device=Zf.device)

        # Sample features from corresponding positions in Zf_with_pos
        grid_coords = []
        for b in range(B):
            batch_coords = []
            for idx in topk_indices[b]:
                # Convert 1D index to 2D coordinates
                idx_h = idx // S_w
                idx_w = idx % S_w
                # Map back to original feature map coordinates (since Fc is downsampled by 2x)
                orig_h = idx_h * 2  # Approximate mapping
                orig_w = idx_w * 2
                orig_h = min(orig_h, H-1)
                orig_w = min(orig_w, W-1)

                # Extract feature from Zf_with_pos
                feature = Zf_with_pos[b, :, orig_h, orig_w]  # (C,)
                Q_topk[b, len(batch_coords)] = feature
                batch_coords.append([idx_h, idx_w])

        # Apply query projection
        Q_topk = self.query_proj(Q_topk)  # (B, K, C)

        # Step 7: Extract candidate coordinates
        Fc_reshaped = Fc.permute(0, 2, 3, 1).reshape(B, S, self.num_vertices_per_building, 2)  # (B, S, Nv, 2)

        Drc_topk = torch.gather(
            Fc_reshaped,
            dim=1,
            index=topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.num_vertices_per_building, 2)
        )  # (B, K, Nv, 2)

        return Q_topk, Drc_topk
