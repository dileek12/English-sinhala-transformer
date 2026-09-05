from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = self.tokenizer_tgt.token_to_id("[SOS]")
        self.eos_token = self.tokenizer_tgt.token_to_id("[EOS]")
        self.pad_token = self.tokenizer_tgt.token_to_id("[PAD]") 
    
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: Any) -> Any:
        src_target_pair = self.ds[idx]
        src_text = src_target_pair[self.src_lang]
        tgt_text = src_target_pair[self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError(f"Sentence length exceeds maximum sequence length of {self.seq_len}")

        # Add SOS and EOS to the source text
        encoder_input = torch.cat(
            [
                torch.tensor([self.sos_token], dtype=torch.int64),
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                torch.tensor([self.eos_token], dtype=torch.int64),
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # Add SOS to decoder input
        decoder_input = torch.cat(
            [
                torch.tensor([self.sos_token], dtype=torch.int64),
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # Add EOS to the label (What we expect the decoder to output)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.eos_token], dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len,)
            "decoder_input": decoder_input,  # (seq_len,)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, 1, seq_len) & (seq_len, seq_len)
            "label": label, # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text
        }

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0