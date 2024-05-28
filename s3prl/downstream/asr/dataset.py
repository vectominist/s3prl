# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ dataset.py ]
#   Synopsis     [ the phone dataset ]
#   Author       [ S3PRL, Xuankai Chang ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import logging
import os
import re


# -------------#
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# -------------#
import torch
from torch.utils.data.dataset import Dataset
import numpy as np

# -------------#
import torchaudio
import soundfile as sf

# -------------#
from .dictionary import Dictionary

SAMPLE_RATE = 16000
HALF_BATCHSIZE_TIME = 2000


####################
# Sequence Dataset #
####################
class SequenceDataset(Dataset):
    def __init__(
        self,
        split,
        bucket_size,
        dictionary,
        libri_root,
        bucket_file,
        dataset_type: str = "librispeech",
        transcription=None,
        **kwargs,
    ):
        super(SequenceDataset, self).__init__()

        self.dictionary = dictionary
        self.libri_root = libri_root
        self.dataset_type = dataset_type.lower()
        self.sample_rate = SAMPLE_RATE
        self.split_sets = kwargs[split]

        # multi speaker
        self.multi_spk = kwargs.get("multi_spk", False)
        if self.multi_spk:
            self.num_spk: int = kwargs.get("num_spk", 2)
            self.spk_overlap: float = kwargs.get("spk_overlap", 0.0)
            print("Multi-speaker setting: ", self.num_spk, self.spk_overlap)

        assert self.dataset_type in {"librispeech", "chime3"}, self.dataset_type

        # Read table for bucketing
        assert os.path.isdir(
            bucket_file
        ), "Please first run `python3 preprocess/generate_len_for_bucket.py -h` to get bucket file."

        # Wavs
        table_list = []
        for item in self.split_sets:
            file_path = os.path.join(bucket_file, item + ".csv")
            if os.path.exists(file_path):
                table_list.append(pd.read_csv(file_path))
            else:
                logging.warning(
                    f"{item} is not found in bucket_file: {bucket_file}, skipping it."
                )

        table_list = pd.concat(table_list)
        table_list = table_list.sort_values(by=["length"], ascending=False)

        X = table_list["file_path"].tolist()
        X_lens = table_list["length"].tolist()

        assert len(X) != 0, f"0 data found for {split}"

        # Transcripts
        if self.dataset_type == "librispeech":
            Y = self._load_transcript(X)
        if self.dataset_type == "chime3":
            Y = self._load_transcript_chime3(X, transcription)

        x_names = set([self._parse_x_name(x) for x in X])
        y_names = set(Y.keys())
        usage_list = list(x_names & y_names)
        assert len(usage_list) > 0, (len(x_names), len(y_names))

        Y = {key: Y[key] for key in usage_list}

        self.X_orig = X
        self.Y = {
            k: self.dictionary.encode_line(v, line_tokenizer=lambda x: x.split()).long()
            for k, v in Y.items()
        }

        # Use bucketing to allow different batch sizes at run time
        self.X = []
        batch_x, batch_len = [], []

        for x, x_len in tqdm(
            zip(X, X_lens),
            total=len(X),
            desc=f"ASR dataset {split}",
            dynamic_ncols=True,
        ):
            if self._parse_x_name(x) in usage_list:
                batch_x.append(x)
                batch_len.append(x_len)

                # Fill in batch_x until batch is full
                if len(batch_x) == bucket_size:
                    # Half the batch size if seq too long
                    if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME):
                        self.X.append(batch_x[: bucket_size // 2])
                        self.X.append(batch_x[bucket_size // 2 :])
                    else:
                        self.X.append(batch_x)
                    batch_x, batch_len = [], []

        # Gather the last batch
        if len(batch_x) > 1:
            if self._parse_x_name(x) in usage_list:
                self.X.append(batch_x)

    def _parse_x_name(self, x):
        return x.split("/")[-1].split(".")[0]

    def _load_wav(self, wav_path):
        # wav, sr = torchaudio.load(os.path.join(self.libri_root, wav_path))
        wav, sr = sf.read(os.path.join(self.libri_root, wav_path))
        wav = torch.from_numpy(wav)
        assert (
            sr == self.sample_rate
        ), f"Sample rate mismatch: real {sr}, config {self.sample_rate}"
        return wav.view(-1)

    def _load_transcript(self, x_list):
        """Load the transcripts for Librispeech"""

        def process_trans(transcript):
            # TODO: support character / bpe
            transcript = transcript.upper()
            return " ".join(list(transcript.replace(" ", "|"))) + " |"

        trsp_sequences = {}
        split_spkr_chap_list = list(set("/".join(x.split("/")[:-1]) for x in x_list))

        for dir in split_spkr_chap_list:
            parts = dir.split("/")
            trans_path = f"{parts[-2]}-{parts[-1]}.trans.txt"
            path = os.path.join(self.libri_root, dir, trans_path)
            assert os.path.exists(path)

            with open(path, "r") as trans_f:
                for line in trans_f:
                    lst = line.strip().split()
                    trsp_sequences[lst[0]] = process_trans(" ".join(lst[1:]))

        return trsp_sequences

    def _load_transcript_chime3(self, x_list, transcription):
        """Load the transcripts for CHiME3"""

        def process_trans(transcript):
            # TODO: support character / bpe
            transcript = transcript.upper()
            transcript = (
                transcript.replace("<NOISE>", "")
                .replace("*", "")
                .replace(".PERIOD", "PERIOD")
                .replace(",COMMA", "COMMA")
                .replace("-HYPHEN", "HYPHEN")
                .replace(":COLON", "COLON")
                .replace("?QUESTION-MARK", "QUESTION MARK")
                .replace("...ELLIPSIS", "ELLIPSIS")
                .replace("-DASH", "DASH")
                .replace(";SEMI-COLON", "SEMI COLON")
                .replace("&AMPERSAND", "AMPERSAND")
                .replace("(LEFT-PAREN", "LEFT PAREN")
                .replace(")RIGHT-PAREN", "RIGHT PAREN")
                .replace('"DOUBLE-QUOTE', "DOUBLE QUOTE")
                .replace("'SINGLE-QUOTE", "SINGLE QUOTE")
                .replace("!EXCLAMATION-POINT", "EXCLAMATION POINT")
            )
            transcript = re.sub(" +", " ", transcript)
            return " ".join(list(transcript.replace(" ", "|"))) + " |"

        trsp_sequences = {}
        with open(transcription, "r") as fp:
            # chime3_et05.trn_all
            for line in fp:
                line = line.strip()
                if line == "":
                    continue
                lst = line.split(" ")
                trsp_sequences[lst[0]] = process_trans(" ".join(lst[1:]))

        return trsp_sequences

    def _build_dictionary(
        self, transcripts, workers=1, threshold=-1, nwords=-1, padding_factor=8
    ):
        d = Dictionary()
        transcript_list = list(transcripts.values())
        Dictionary.add_transcripts_to_dictionary(transcript_list, d, workers)
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Load acoustic feature and pad
        wav_batch = [self._load_wav(x_file).numpy() for x_file in self.X[index]]
        label_batch = [
            self.Y[self._parse_x_name(x_file)].numpy() for x_file in self.X[index]
        ]
        filename_batch = [Path(x_file).stem for x_file in self.X[index]]

        if self.multi_spk:
            # NOTE: chime only
            wav_batch_new = []
            label_batch_new = []
            for i, x_file in enumerate(self.X[index]):
                add_x_file = []
                x_spk = self._parse_x_name(x_file).split("_")[0]
                for _ in range(self.num_spk - 1):
                    while True:
                        add_index = np.random.randint(len(self.X_orig))
                        spk = self._parse_x_name(self.X_orig[add_index]).split("_")[0]
                        if x_spk != spk:
                            break
                    add_x_file.append(self.X_orig[add_index])

                add_wav = [self._load_wav(_x_file).numpy() for _x_file in add_x_file]
                add_label = [
                    self.Y[self._parse_x_name(_x_file)].numpy()
                    for _x_file in add_x_file
                ]

                if self.spk_overlap < 0.01:
                    wav = np.concatenate([wav_batch[i]] + add_wav, axis=0)
                else:
                    all_lens = [len(wav_batch[i])] + [len(w) for w in add_wav]
                    min_len = min(all_lens)
                    overlap_len = int(min_len * self.spk_overlap)
                    total_len = sum(all_lens) - overlap_len * len(add_wav)

                    wav = np.zeros(total_len, dtype=wav_batch[i].dtype)
                    wav[: len(wav_batch[i])] = wav_batch[i]
                    start = len(wav_batch[i]) - overlap_len
                    for j in range(len(add_wav)):
                        wav[start : start + len(add_wav[j])] = add_wav[j]
                        start += len(add_wav[j]) - overlap_len

                label = [label_batch[i]] + add_label
                label = [label[j][:-1] for j in range(len(label) - 1)] + [
                    label[-1]
                ]  # remove <eos>
                label = np.concatenate(label, axis=0)

                wav_batch_new.append(wav)
                label_batch_new.append(label)

            wav_batch = wav_batch_new
            label_batch = label_batch_new

        return (
            wav_batch,
            label_batch,
            filename_batch,
        )  # bucketing, return ((wavs, labels))

    def collate_fn(self, items):
        assert len(items) == 1
        return (
            items[0][0],
            items[0][1],
            items[0][2],
        )  # hack bucketing, return (wavs, labels, filenames)
