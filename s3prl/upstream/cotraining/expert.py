from argparse import Namespace
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ..interfaces import UpstreamBase
from .fbank import compute_fbank
from .model import Cotraining


logger = logging.getLogger(__name__)


class UpstreamExpert(UpstreamBase):
    def __init__(
        self,
        ckpt,
        feat_select: str = "feats",
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.model = Cotraining()
        self.model.load(ckpt)

        self.feat_select = feat_select
        assert feat_select in {
            "feats",
            "vq"
        }, feat_select

        logger.info(f"Feature selection: {feat_select}")

    def get_downsample_rates(self, key: str) -> int:
        return 160

    def forward(self, wavs):
        feats, feat_len, mask = compute_fbank(wavs, cmvn=True)

        if self.feat_select == "feats":
            _, features, _ = self.model.forward_pred(feats, feat_len, mask)
            return {"hidden_states": features, "last_hidden_state": features[-1]}
        else:
            _, results, _ = self.model.forward_conf(feats, mask)
            return {"hidden_states": [results["latent_probs"].argmax(-1)]}
