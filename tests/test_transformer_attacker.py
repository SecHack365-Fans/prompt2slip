import os
from typing import Tuple
import json
import logging

import pytest
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import torch
from prompt2slip import CLMAttacker

texts = []
with open(
    os.path.join(os.path.dirname(__file__), "text4test.txt"), "r", encoding="utf-8"
) as f:
    for line in f:
        texts.append(line.rstrip("\n"))
target_words = ["apple", "orange", "seed", "upper", "worst"]

params = [
    ("AdamW", 0.3, 2, 1.0, 1.0, texts, target_words, False),
    ("AdamW", 0.3, 2, 1.0, 1.0, texts, target_words, True),
    ("AdamW", 1.0, 2, 1.0, 1.0, texts, target_words, False),
    ("AdamW", 0.5, 4, 0.5, 1.0, texts, target_words, True),
    ("AdamW", 0.4, 6, 0.01, 0.1, texts, target_words, False),
]


@pytest.mark.parametrize(
    "optimizer, lr, n_iter, weight_cossim, weight_fluency, texts, target_words, put_target",
    params,
)
def test_clm_attack_0(
    optimizer,
    lr,
    n_iter,
    weight_cossim,
    weight_fluency,
    texts,
    target_words,
    put_target,
):
    n_samples = 2
    from transformers import GPT2Tokenizer, GPT2LMHeadModel

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    target_ids = torch.Tensor(tokenizer.convert_tokens_to_ids(target_words)).long()

    attacker = CLMAttacker(
        victim_transformer=model,
        victim_tokenizer=tokenizer,
        refer_transformer=None,
        refer_tokenizer=None,
        optimizer=optimizer,
        init_val=15,
        lr=lr,
        grad_clip_max=10,
        grad_clip_norm=1000,
        margin=0.1,
        weight_cossim=weight_cossim,
        weight_fluency=weight_fluency,
        log_level=logging.DEBUG,
    )
    if put_target:
        forbit_mask = attacker.make_forbit_mask_from_texts(texts)
        forbit_mask = torch.Tensor(forbit_mask).bool()
    else:
        forbit_mask = None
    output = attacker.attack_by_text(
        texts,
        target=target_ids,
        forbid_mask=None,
        n_itter=n_iter,
        n_sample=n_samples,
        target_mask=forbit_mask,
    )
    assert len(output[0]) == n_samples

    for i, out in enumerate(output[0]):
        assert len(out) == len(texts)
        for j, bout in enumerate(out):
            print(
                f"{i}-{j} ; ",
                " ".join(
                    [
                        token.lstrip("Ġ ")
                        for token in bout
                        if token != tokenizer.eos_token
                    ]
                ),
            )
