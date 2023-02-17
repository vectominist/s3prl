import logging
import sys

import torch
from torch.nn.utils.rnn import pad_sequence

from ..interfaces import UpstreamBase

SAMPLE_RATE = 16000
EXAMPLE_SEC = 5

logger = logging.getLogger(__name__)

sys.path.append("/data/sls/r/u/hengjui/home/scratch/speech-disentangle")


class UpstreamExpert(UpstreamBase):
    def __init__(
        self,
        ckpt,
        feat_select: str = "all_feats",
        **kwargs,
    ):
        super().__init__(**kwargs)

        from src.model import BYOLModel

        self.model = BYOLModel.load_from_checkpoint(ckpt, strict=False)
        if self.model.encoder_type == "HuBERT":
            self.model.encoder.model.feature_grad_mult = 0.0
        self.feat_select = feat_select

        assert self.feat_select in {
            "all_feats",
            "hidden",
            "disentangled",
        }, self.feat_select

        logger.info(f"Feature selection: {self.feat_select}")

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs):
        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        feat, _, feat_list, _ = self.model(
            (padded_wav, wav_lengths, wav_padding_mask), feat_only=True, apply_aug=False
        )

        outputs = {}
        if self.feat_select == "all_feats":
            outputs["hidden_states"] = [feat] + feat_list
        elif self.feat_select == "hidden":
            outputs["hidden_states"] = feat_list
        elif self.feat_select == "disentangled":
            outputs["hidden_states"] = [feat]

        outputs["last_hidden_state"] = outputs["hidden_states"][-1]

        return outputs
