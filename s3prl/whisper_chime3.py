import argparse

import torch
import whisper
from tqdm import tqdm
from whisper.normalizers import EnglishTextNormalizer

from s3prl.metric import cer, wer
from s3prl.util.seed import fix_random_seeds
from s3prl.downstream.asr.dataset import SequenceDataset, Dictionary


def token_to_word(text):
    # Hard coding but it is only used here for now.
    # Assumption that units are characters. Doesn't handle BPE.
    # Inter-character separator is " " and inter-word separator is "|".
    return text.replace(" ", "").replace("|", " ").strip()


class CHiME3(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        bucket_file,
        transcription,
        split="et05_bus_real",
        dictionary="./downstream/asr/char.dict",
    ):
        self.dictionary = Dictionary.load(dictionary)
        kwargs = {split: [split]}
        self.dataset = SequenceDataset(
            split,
            1,
            self.dictionary,
            root,
            bucket_file,
            dataset_type="chime3",
            transcription=transcription,
            **kwargs
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        wav, label, filename = self.dataset[item]

        audio = whisper.pad_or_trim(wav[0].flatten())
        mel = whisper.log_mel_spectrogram(audio)

        uid = filename[0]
        label = label[0]
        label_idx = (label != self.dictionary.pad()) & (label != self.dictionary.eos())
        target_token_ids = label[label_idx].tolist()
        target_tokens = self.dictionary.string(target_token_ids)
        text = token_to_word(target_tokens)

        return (uid, mel, text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="base.en")
    parser.add_argument(
        "--chime_dir",
        default="/data/sls/d/corpora/original/chime3_Aug15_2022/CHiME3/data/audio/16kHz/isolated_1ch_track",
    )
    parser.add_argument(
        "--bucket_file",
        default="/data/sls/r/u/hengjui/home/scratch/dataset/chime3_util/len_for_bucket",
    )
    parser.add_argument(
        "--transcription",
        default="/data/sls/r/u/hengjui/home/scratch/dataset/chime3_util/chime3.trn_all",
    )
    parser.add_argument("--split", default="et05_bus_real")

    parser.add_argument("--out", default="./result/whisper_asr_chime3")
    parser.add_argument("--njobs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    fix_random_seeds()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = whisper.load_model(args.model_name, device=device)
    normalizer = EnglishTextNormalizer()
    options = whisper.DecodingOptions(language="english", without_timestamps=True)

    dataset = CHiME3(
        args.chime_dir,
        args.bucket_file,
        args.transcription,
        args.split,
    )
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
