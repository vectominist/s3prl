from argparse import Namespace
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ..interfaces import UpstreamBase
from .model import extract_feat

SAMPLE_RATE = 16000
EXAMPLE_SEC = 5

logger = logging.getLogger(__name__)


class UpstreamExpert(UpstreamBase):
    def __init__(
        self,
        ckpt,
        get_teacher: bool = False,
        get_vq: bool = False,
        get_pred: bool = False,
        feat_select: str = "student_feats",
        **kwargs,
    ):
        super().__init__(**kwargs)
        import fairseq

        cfg = Namespace()
        cfg.common = Namespace()
        cfg.common.user_dir = "/data/sls/u/alexhliu/home/ssl-dev/examples/data2vec"
        fairseq.utils.import_user_module(cfg.common)
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt])
        self.model = model[0]
        self.task = task

        if len(feat_select.split("-")) == 2:
            self.feat_select, self.feat_layer = feat_select.split("-")
            self.feat_layer = int(self.feat_layer)
        else:
            self.feat_select = feat_select
            self.feat_layer = -1
        self.get_teacher = get_teacher or (self.feat_select == "teacher_feats")
        self.get_vq = get_vq or (self.feat_select in {"vq_onehot", "vq_labels"})
        self.get_pred = get_pred or (
            self.feat_select in {"student_logit", "student_prob"}
        )

        assert self.feat_select in {
            "student_feats",
            "teacher_feats",
            "vq_onehot",
            "vq_labels",
            "student_logit",
            "student_prob",
        }, self.feat_select

        logger.info(
            f"Feature selection: {self.feat_select} (layer = {self.feat_layer})"
        )
        logger.info(
            f"Returns teacher = {self.get_teacher} , VQ = {self.get_vq} , pred = {self.get_pred}"
        )

        self.model.feature_grad_mult = 0.0
        self.model.encoder.layerdrop = 0.0

        # if not self.get_teacher and not self.get_vq:
        #     del self.model.ema

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs):
        if self.task.cfg.normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        results = extract_feat(
            self.model,
            padded_wav,
            padding_mask=wav_padding_mask,
            get_teacher=self.get_teacher,
            get_vq=self.get_vq,
            get_pred=self.get_pred,
        )

        outputs = {}

        if self.get_teacher or self.get_vq:
            outputs["teacher_states"] = results["teacher_feats"]
        if self.get_vq:
            outputs["vq_states"] = results["vq_onehot"]
            outputs["vq_labels"] = results["vq_labels"]
        if self.get_pred:
            outputs["student_logit"] = results["student_logit"]
            outputs["student_prob"] = results["student_prob"]

        outputs["hidden_states"] = results[self.feat_select]
        outputs["last_hidden_state"] = outputs["hidden_states"][-1]

        if self.feat_layer >= 0:
            outputs["hidden_states"] = [outputs["hidden_states"][self.feat_layer]]

        return outputs
