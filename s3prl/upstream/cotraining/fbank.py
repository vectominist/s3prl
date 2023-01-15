from typing import List

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence


EPS = 1e-10


def compute_fbank(wavs: List[torch.Tensor], cmvn: bool = True):
    feats = []
    feat_len = []

    with torch.no_grad():
        for wav in wavs:
            feat = torchaudio.compliance.kaldi.fbank(wav.unsqueeze(0), num_mel_bins=40)
            feats.append(feat)
            feat_len.append(len(feat))

        if cmvn:
            feat_cat = torch.cat(feats, dim=0)
            std, mean = torch.std_mean(feat_cat, 0)
        feats = pad_sequence(feats, batch_first=True)
        # (B, T, 40)

        if cmvn:
            feats = (feats - mean.view(1, 1, -1)) / (EPS + std.view(1, 1, -1))

    max_len = max(feat_len)
    feat_len = torch.LongTensor(feat_len).to(feats.device)
    ids = torch.arange(max_len).unsqueeze(0).to(feats.device)
    mask = ids >= feat_len.unsqueeze(1)

    return feats, feat_len, mask
