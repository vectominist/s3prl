import logging
import sys

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ..interfaces import UpstreamBase

SAMPLE_RATE = 16000
EXAMPLE_SEC = 5

logger = logging.getLogger(__name__)


def convert_state_dict(state_dict: dict) -> dict:
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v
    return new_state_dict


class UpstreamExpert(UpstreamBase):
    def __init__(
        self,
        ckpt,
        feat_select: str = "hidden",
        **kwargs,
    ):
        super().__init__(**kwargs)

        sys.path.append("/data/sls/r/u/hengjui/home/scratch/speech-disentangle")
        from src.nn import Disentangler

        ckpt = torch.load(ckpt, map_location="cpu")
        config = ckpt["hyper_parameters"]["model"]["core"]
        config["decoder_out_dim"] = 80
        state_dict = convert_state_dict(ckpt["state_dict"])

        self.model = Disentangler(**config)
        self.model.load_state_dict(state_dict)

        self.feat_select = feat_select

        assert self.feat_select in {
            "raw",
            "hidden",
            "all_hidden",
            "content",
            "other",
            "vq",
            "logits",
            "probs",
            "codes",
            "spectrogram",
        }, self.feat_select

        self.model.normalize_codebook()

        logger.info(f"Feature selection: {self.feat_select}")

    def get_downsample_rates(self, key: str) -> int:
        if self.feat_select in {"spectrogram"}:
            return 160
        else:
            return 320

    def forward(self, wavs):
        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        wavs = pad_sequence(wavs, batch_first=True)

        outputs = {}

        if self.feat_select in {"raw"}:
            results = self.model(wavs, padding_mask, is_train=False)
            outputs["hidden_states"] = results["feat_list"]
            encoder_dim = results["feat_list"][-1].size(-1)

            if self.model.content_repr_dim == encoder_dim:
                outputs["hidden_states"].append(results["content_repr"])
            if self.model.other_repr_dim == encoder_dim:
                outputs["hidden_states"].append(results["other_repr"])

            if self.feat_select == "raw":
                outputs = {**outputs, **results}

        elif self.feat_select == "hidden":
            # Hidden representations of the encoder
            feat_list, _, _ = self.model.forward_encoder(wavs, padding_mask)
            outputs["hidden_states"] = feat_list
            # [(B, T, encoder_dim) FloatTensor] x num_encoder_layers

        elif self.feat_select == "all_hidden":
            # Hidden representations of the encoder + disentangled representations
            feat_list, content_repr, other_repr, _, _ = (
                self.model.get_disentangled_repr(wavs, padding_mask)
            )
            outputs["hidden_states"] = results["feat_list"]
            encoder_dim = results["feat_list"][-1].size(-1)

            if self.model.content_repr_dim == encoder_dim:
                outputs["hidden_states"].append(results["content_repr"])
            if self.model.other_repr_dim == encoder_dim:
                outputs["hidden_states"].append(results["other_repr"])

        elif self.feat_select == "content":
            # Disentangled content representations
            content_repr, _, _ = self.model.get_content_repr(wavs, padding_mask)
            outputs["hidden_states"] = [content_repr]
            # [(B, T, content_repr_dim) FloatTensor]

        elif self.feat_select == "other":
            # Disentangled other representations (e.g., speaker, noise, etc.)
            other_repr, _, _ = self.model.get_other_repr(wavs, padding_mask)
            outputs["hidden_states"] = [other_repr]
            # [(B, T, other_repr_dim) FloatTensor]

        elif self.feat_select in {"vq", "logits", "probs", "codes"}:
            vq_repr, logits_list, codes_list, _, _ = self.model.get_rvq(
                wavs, padding_mask
            )

            if self.feat_select == "vq":
                # Content representations after VQ
                outputs["hidden_states"] = [vq_repr]
                # [(B, T, content_repr_dim) FloatTensor]
            elif self.feat_select == "logits":
                # Logits of each codebook
                outputs["hidden_states"] = logits_list
                # [(B, T, codebook_size) FloatTensor] x num_codebooks
            elif self.feat_select == "probs":
                # Softmax of the logits of each codebook
                outputs["hidden_states"] = [
                    F.softmax(logits_list[i], dim=2) for i in range(len(logits_list))
                ]
                # [(B, T, codebook_size) FloatTensor] x num_codebooks
            elif self.feat_select == "codes":
                # Codes of each codebook
                outputs["hidden_states"] = [torch.stack(codes_list, dim=2)]
                # [(B, T, num_codebooks) LongTensor]

        elif self.feat_select == "spectrogram":
            # Reconstructed spectrogram
            spectrogram, _, _ = self.model.get_recontruction(wavs, padding_mask)
            outputs["hidden_states"] = [spectrogram]

        outputs["last_hidden_state"] = outputs["hidden_states"][-1]

        return outputs
