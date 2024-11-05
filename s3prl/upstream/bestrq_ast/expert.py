from argparse import Namespace
import logging
import sys

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ..interfaces import UpstreamBase

SAMPLE_RATE = 16000
EXAMPLE_SEC = 5

logger = logging.getLogger(__name__)

sys.path.append("/data/sls/scratch/hengjui/fairseq-bestrq/fairseq/models")

from bestrq_ast.backbone import BestRqAst


class UpstreamExpert(UpstreamBase):
    def __init__(
        self,
        ckpt,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model = BestRqAst.load_from_checkpoint(ckpt)
        self.normalize_wav = self.model.full_config["task"]["normalize"]

        self.model.encoder.layerdrop = 0.0

    def get_downsample_rates(self, key: str) -> int:
        return self.model.frame_downsample_rate

    def forward(self, wavs):
        if self.normalize_wav:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        result = self.model.get_frame_features(
            padded_wav, padding_mask=wav_padding_mask
        )

        outputs = {}
        outputs["hidden_states"] = [result["features"]] + [
            h[0] for h in result["layer_results"]
        ]
        outputs["last_hidden_state"] = outputs["hidden_states"][-1]

        return outputs
