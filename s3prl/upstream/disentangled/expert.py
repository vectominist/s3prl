import logging

import torch
from torch.nn.utils.rnn import pad_sequence

from ..interfaces import UpstreamBase
from .model import load_pretrained

logger = logging.getLogger(__name__)


class UpstreamExpert(UpstreamBase):
    def __init__(
        self,
        ckpt,
        upstream_name: str,
        upstream_layer: int,
        upstream_ckpt: str = None,
        feature_selection: str = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        from s3prl.nn import S3PRLUpstream

        self.upstream = S3PRLUpstream(upstream_name, upstream_ckpt)
        self.layer = int(upstream_layer)

        self.model = load_pretrained(ckpt)

        assert feature_selection in {"cs", "c", "s"}, feature_selection
        self.feature_selection = feature_selection

    def get_downsample_rates(self, key: str) -> int:
        return self.upstream.downsample_rates

    def forward(self, wavs):
        device = wavs[0].device
        wav_lens = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wavs = pad_sequence(wavs, batch_first=True)
        upstream_out, _ = self.upstream(wavs, wav_lens)

        feat = upstream_out[self.layer]
        cs, c, s = self.model.disentangle(feat)

        hidden_states = None
        if self.feature_selection == "c":
            hidden_states = [c]
        elif self.feature_selection == "s":
            hidden_states = [s]
        else:
            hidden_states = [cs]

        return {"hidden_states": hidden_states}
