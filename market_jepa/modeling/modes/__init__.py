"""Training modes (LeJEPA, I-JEPA, MAE, CPC, DINO, BYOL, TS2Vec, CoST, TF-C,
TimeMAE, Supervised, FinanceBaseline, PretrainedTSFM)."""

from .base import TrainingModel

from .lejepa import SIGReg, LeJEPA
from .ijepa import IJEPA, IJEPAPredictor
from .mae import MAE, MAEDecoder
from .cpc import CPC, CPCPredictiveHeads
from .dino import DINO, DINOHead
from .byol import BYOL
from .ts2vec import TS2Vec
from .cost import CoST
from .tfc import TFC
from .timemae import TimeMAE
from .supervised import SupervisedModel, MultiTaskSupervisedModel
from .finance_baselines import FinanceBaseline
from .finance_baseline_params import BaselineParams, MonthlyBaselineParams
from .pretrained_tsfm import PretrainedTSFM

__all__ = [
    "TrainingModel",
    "SIGReg",
    "LeJEPA",
    "IJEPA",
    "IJEPAPredictor",
    "MAE",
    "MAEDecoder",
    "CPC",
    "CPCPredictiveHeads",
    "DINO",
    "DINOHead",
    "BYOL",
    "TS2Vec",
    "CoST",
    "TFC",
    "TimeMAE",
    "SupervisedModel",
    "MultiTaskSupervisedModel",
    "FinanceBaseline",
    "BaselineParams",
    "MonthlyBaselineParams",
    "PretrainedTSFM",
]
