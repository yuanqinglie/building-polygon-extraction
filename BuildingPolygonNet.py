

import torchvision.models as models

class BuildingPolygonNet(nn.Module):
    def __init__(self,
                 encoder_type: str = "resnet50",
                 encoder_channels: List[int] = [256, 512, 1024, 2048],  # ResNet50 feature layer channels
                 d_model: int = 256,
                 num_queries: int = 100,
                 num_vertices: int = 8,
                 top_e: int = 16,
                 pretrained: bool = True):
        super().__init__()

        self.encoder_type = encoder_type
        self.encoder_channels = encoder_channels

        # Encoder initialization
        self.encoder = self._build_encoder(encoder_type, pretrained)

        # Feature adaptation layers - adjust encoder output channels to d_model
        self.adapt_l2 = nn.Conv2d(encoder_channels[0], d_model, 1) if encoder_channels[0] != d_model else nn.Identity()
        self.adapt_l3 = nn.Conv2d(encoder_channels[1], d_model, 1) if encoder_channels[1] != d_model else nn.Identity()
        self.adapt_l4 = nn.Conv2d(encoder_channels[2], d_model, 1) if encoder_channels[2] != d_model else nn.Identity()
        self.adapt_l5 = nn.Conv2d(encoder_channels[3], d_model, 1) if encoder_channels[3] != d_model else nn.Identity()

        # Core modules - update input channels to d_model
        self.cfpe = CFPE(
            in_channels_l2=d_model,  # Use adapted channel count
            in_channels_l3=d_model,
            in_channels_l4=d_model,
            in_channels_l5=d_model,
            out_channels=d_model
        )

        self.dqo = DQO(
            in_channels=d_model,
            num_queries=num_queries,
            num_vertices_per_building=num_vertices
        )

        # MVDA layers
        self.mvda_layers = nn.ModuleList([
            MVDA(d_model=d_model, n_heads=8, num_vertices_per_building=num_vertices)
            for _ in range(2)
        ])

        self.ectr = ECTR(
            d_model=d_model,
            num_vertices_per_building=num_vertices,
            top_e=min(top_e, num_vertices * num_vertices)
        )

        # Output head
        self.vertex_confidence = nn.Linear(d_model, 1)

    def _build_encoder(self, encoder_type: str, pretrained: bool) -> nn.Module:
        """Build encoder"""
        if encoder_type == "resnet50":
            return ResNet50Encoder(pretrained=pretrained)
        elif encoder_type == "resnet101":
            return ResNet101Encoder(pretrained=pretrained)
        elif encoder_type == "efficientnet":
            return EfficientNetEncoder(pretrained=pretrained)
        elif encoder_type == "simple":
            return SimpleEncoder(self.encoder_channels)
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encoder forward pass
        L2, L3, L4, L5 = self.encoder(x)

        # Adapt feature channel counts
        L2_adapted = self.adapt_l2(L2)
        L3_adapted = self.adapt_l3(L3)
        L4_adapted = self.adapt_l4(L4)
        L5_adapted = self.adapt_l5(L5)

        # CFPE module
        Zf = self.cfpe(L2_adapted, L3_adapted, L4_adapted, L5_adapted)

        # DQO module - use adapted L4
        Q, Drc = self.dqo(Zf, L4_adapted)

        # MVDA layer iterative optimization
        for mvda in self.mvda_layers:
            Q, Drc = mvda(Q, Drc, Zf, L4_adapted)

        # ECTR module
        SA = self.ectr(Q, Drc, Zf)

        # Vertex confidence prediction
        vertices_feat = Q.unsqueeze(2).expand(-1, -1, self.ectr.num_vertices, -1)
        confidence = torch.sigmoid(self.vertex_confidence(vertices_feat))

        return Drc, SA, confidence

