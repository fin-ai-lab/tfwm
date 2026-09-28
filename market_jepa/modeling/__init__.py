"""Time Series JEPA models with interchangeable backbones.

This module provides a modular JEPA (Joint Embedding Predictive Architecture)
framework for time series self-supervised learning, using SIGReg loss from
LeJEPA (Balestriero & LeCun, 2025).

Example usage:
    ```python
    from modeling import LeJEPA, TransformerBackbone, create_backbone

    backbone = create_backbone(
        backbone_type="transformer",
        n_features=10,
        d_embedding=512,
    )
    model = LeJEPA(backbone, proj_dim=128, lamb=0.02)

    # Forward pass
    # x shape: (batch, n_views, n_features, length)
    # or list of tensors for variable-length views
    output = model(x, return_loss=True)
    embeddings = output["embeddings"]  # (batch, n_views, d_embedding)
    loss = output["lejepa_loss"]
    ```
"""

# Core components
from .modes.utils import RMSNorm
from .modes import TrainingModel

# Backbones
from .backbones import (
    TimeSeriesBackbone,
    create_backbone,
    TransformerBackbone,
    TransformerConfig,
    PatchEmbedding1D,
    ViTBlock,
    ResNetBackbone,
    BasicBlock1D,
    BottleneckBlock1D,
    EfficientNetBackbone,
    MBConv1D,
    FusedMBConv1D,
    ConvNeXtBackbone,
    CNBlock1D,
    InceptionBackbone,
    BasicConv1d,
    InceptionA1D,
    InceptionB1D,
    InceptionC1D,
    InceptionD1D,
    InceptionE1D,
    PatchTSTBackbone,
    PatchTSTConfig,
)

# Modes
from .modes import (
    SIGReg,
    LeJEPA,
    IJEPA,
    IJEPAPredictor,
    MAE,
    MAEDecoder,
    CPC,
    CPCPredictiveHeads,
    TS2Vec,
    CoST,
    TFC,
    TimeMAE,
    SupervisedModel,
)

__all__ = [
    # Core
    "RMSNorm",
    "TimeSeriesBackbone",
    "TrainingModel",
    "create_backbone",
    # Modes
    "SIGReg",
    "LeJEPA",
    "IJEPA",
    "IJEPAPredictor",
    "MAE",
    "MAEDecoder",
    "CPC",
    "CPCPredictiveHeads",
    "TS2Vec",
    "CoST",
    "TFC",
    "TimeMAE",
    "SupervisedModel",
    # Transformer
    "TransformerBackbone",
    "TransformerConfig",
    "PatchEmbedding1D",
    "ViTBlock",
    # ResNet
    "ResNetBackbone",
    "BasicBlock1D",
    "BottleneckBlock1D",
    # EfficientNet
    "EfficientNetBackbone",
    "MBConv1D",
    "FusedMBConv1D",
    # ConvNeXt
    "ConvNeXtBackbone",
    "CNBlock1D",
    # Inception
    "InceptionBackbone",
    "BasicConv1d",
    "InceptionA1D",
    "InceptionB1D",
    "InceptionC1D",
    "InceptionD1D",
    "InceptionE1D",
    # PatchTST
    "PatchTSTBackbone",
    "PatchTSTConfig",
]
