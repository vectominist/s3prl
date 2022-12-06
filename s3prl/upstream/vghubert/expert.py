import logging
import os
import pickle

import torch
from torch.nn.utils.rnn import pad_sequence


from ..interfaces import UpstreamBase
from .encoder import AudioEncoder

logger = logging.getLogger(__name__)


class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, feature_selection: str = None, **kwargs):
        super().__init__(**kwargs)
        with open(os.path.join(ckpt, "args.pkl"), "rb") as f:
            model_args = pickle.load(f)
        model = AudioEncoder(model_args)
        bundle = torch.load(os.path.join(ckpt, "best_bundle.pth"))
        model.carefully_load_state_dict(bundle["dual_encoder"], load_all=True)

        self.model = model

        self.model.feature_grad_mult = 0.0
        self.model.encoder.layerdrop = 0.0

        self.feature_selection = feature_selection

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs):
        device = wavs[0].device
        # if self.wav_normalize:
        #     if self.numpy_wav_normalize:
        #         wavs = zero_mean_unit_var_norm([wav.cpu().numpy() for wav in wavs])
        #         wavs = [torch.from_numpy(wav).to(device) for wav in wavs]
        #     else:
        #         wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        results = self.model(padded_wav, wav_padding_mask, superb=True)

        return {"hidden_states": results}
