import argparse
from pathlib import Path

import torch
import torchaudio
import whisper
from tqdm import tqdm
from whisper.normalizers import EnglishTextNormalizer

from s3prl.downstream.audio_aug import AudioAugmentation
from s3prl.metric import cer, wer
from s3prl.util.seed import fix_random_seeds


class LibriSpeech(torch.utils.data.Dataset):
    def __init__(
        self, root, augmenter: AudioAugmentation, split="test-clean"
    ):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=root,
            url=split,
            download=False,
        )
        self.augmenter = augmenter

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio, sample_rate, text, speaker, chapter, utt = self.dataset[item]
        assert sample_rate == 16000

        audio = whisper.pad_or_trim(audio.flatten())
        audio = self.augmenter(audio.unsqueeze(0)).squeeze(0)
        mel = whisper.log_mel_spectrogram(audio)

        uid = f"{speaker}-{chapter}-{str(utt).zfill(4)}"

        return (uid, mel, text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="base.en")
    parser.add_argument(
        "--libri_dir", default="/data/sls/temp/alexhliu/data/LibriSpeech"
    )
    parser.add_argument("--libri_split", default="test-clean")

    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--bg", action="store_true")
    parser.add_argument(
        "--bg_path", default="/data/sls/r/u/hengjui/home/scratch/dataset/musan"
    )
    parser.add_argument("--snr", type=float, default=0.0)
    parser.add_argument("--ir", action="store_true")
    parser.add_argument(
        "--ir_path",
        default="/data/sls/scratch/nauman/datasets/rirs/RIRS_NOISES/simulated_rirs",
    )
    parser.add_argument("--out", default="./result/whisper_asr")
    parser.add_argument("--njobs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    fix_random_seeds()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = whisper.load_model(args.model_name, device=device)
    normalizer = EnglishTextNormalizer()
    options = whisper.DecodingOptions(language="english", without_timestamps=True)

    augmenter = AudioAugmentation(
        noise_prob=1.0 if args.noise else 0.0,
        noise_range=[args.snr, args.snr],
        background_noise_prob=1.0 if args.bg else 0.0,
        background_noise_range=[0.0, 20.0],
        background_noise_path=args.bg_path,
        ir_prob=1.0 if args.ir else 0.0,
        ir_path=[args.ir_path],
    )

    dataset = LibriSpeech(args.libri_dir, augmenter, "test-clean")
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.njobs
    )

    id_list = []
    hyp_list = []
    ref_list = []

    for uids, mels, refs in tqdm(loader):
        with torch.no_grad():
            results = model.decode(mels.to(device), options)
            id_list.extend(uids)
            hyp_list.extend([res.text for res in results])
            ref_list.extend(refs)

    hyp_list = [normalizer(hyp) for hyp in hyp_list]
    ref_list = [normalizer(ref) for ref in ref_list]

    # Compute WER / CER
    print(f"WER = {wer(hyp_list, ref_list) * 100.0} %")
    print(f"CER = {cer(hyp_list, ref_list) * 100.0} %")
    print()

    # Save results
    with open(args.out, "w") as fp:
        for id, hyp in zip(id_list, hyp_list):
            fp.write(f"{id} {hyp}\n")


if __name__ == "__main__":
    main()
