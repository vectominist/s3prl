import logging
from pathlib import Path
from typing import List, Tuple, Union
import os

import numpy as np
import torch
from torch_audiomentations import (
    AddBackgroundNoise,
    AddColoredNoise,
    ApplyImpulseResponse,
    Compose,
    Mix,
    PitchShift,
)

logger = logging.getLogger("audio_aug")


class AudioAugmentation:
    def __init__(
        self,
        pitch_shift_prob: float = 0.0,
        pitch_shift_range: Tuple[float, float] = (-4.0, -4.0),
        mix_prob: float = 0.0,
        mix_range: Tuple[float, float] = (5.0, 15.0),
        noise_prob: float = 0.0,
        noise_range: Tuple[float, float] = (5.0, 40.0),
        background_noise_prob: float = 0.0,
        background_noise_path: Union[List[Path], List[str], Path, str] = None,
        background_noise_range: Tuple[float, float] = (5.0, 30.0),
        ir_prob: float = 0.0,
        ir_path: Union[List[Path], List[str], Path, str] = None,
        sample_rate: int = 16000,
    ) -> None:

        self.sample_rate = sample_rate

        self.pitch_shift_prob = pitch_shift_prob
        self.mix_prob = mix_prob
        self.noise_prob = noise_prob
        self.background_noise_prob = background_noise_prob

        self.aug_probs = []
        self.transforms = []

        if pitch_shift_prob > 0.0:
            print(f"Apply pitch shift with range {pitch_shift_range}")
            self.aug_probs.append(pitch_shift_prob)
            self.transforms.append(
                Compose(
                    [
                        PitchShift(
                            pitch_shift_range[0],
                            pitch_shift_range[1],
                            p=1.0,
                        )
                    ]
                )
            )
        if ir_path is not None and ir_prob > 0.0:
            print(f"Apply impulse response")
            print(f"IR source: {ir_path}")
            self.aug_probs.append(ir_prob)
            ir_path_list = []
            for path in ir_path:
                if os.path.isdir(path):
                    path = Path(path)
                    if path.stem == "real_rirs_isotropic_noises":
                        files_1 = path.rglob("*_rir_*.wav")
                        files_2 = path.rglob("*_air_*.wav")
                        files = list(files_1) + list(files_2)
                    else:
                        files = path.rglob("*.wav")
                    files = [str(f) for f in files]
                    ir_path_list += files
                    logger.info(f"Found {len(files)} audio files in {str(path)}")
                else:
                    raise NotImplementedError
            self.transforms.append(
                Compose(
                    [
                        ApplyImpulseResponse(
                            ir_path_list,
                            p=1.0,
                            sample_rate=sample_rate,
                            compensate_for_propagation_delay=True,
                        )
                    ]
                )
            )
        if mix_prob > 0.0:
            print(f"Apply utterance mix with range {mix_range}")
            self.aug_probs.append(mix_prob)
            self.transforms.append(Compose([Mix(mix_range[0], mix_range[1], p=1.0)]))
        if noise_prob > 0.0:
            print(f"Apply Gaussian noise with range {noise_range}")
            self.aug_probs.append(noise_prob)
            self.transforms.append(
                Compose(
                    [
                        AddColoredNoise(
                            noise_range[0],
                            noise_range[1],
                            min_f_decay=0.0,
                            max_f_decay=0.0,
                            p=1.0,
                        )
                    ]
                )
            )
        if background_noise_path is not None and background_noise_prob > 0.0:
            print(f"Apply background noise with range {background_noise_range}")
            print(f"Noise source: {background_noise_path}")
            self.aug_probs.append(background_noise_prob)
            self.transforms.append(
                Compose(
                    [
                        AddBackgroundNoise(
                            background_noise_path,
                            background_noise_range[0],
                            background_noise_range[1],
                            min_len_ratio=1.0,
                            repeat_same=True,
                            p=1.0,
                            sample_rate=sample_rate,
                        )
                    ]
                )
            )

        self.num_augs = len(self.aug_probs)

        print(f"Augmentation probs: {self.aug_probs}")

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (Batch, Time)
        wav = wav.unsqueeze(1)  # (Batch, 1, Time)
        T = wav.shape[2]

        if self.num_augs > 0:
            with torch.no_grad():
                i = np.random.choice(self.num_augs, p=self.aug_probs)
                wav = self.transforms[i](wav, sample_rate=self.sample_rate)

        return wav.squeeze(1).clamp(-1.0, 1.0)
