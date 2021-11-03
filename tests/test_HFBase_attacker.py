import os
import logging
import pytest
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from prompt2slip import HFBaseAttacker

texts = []
with open(
    os.path.join(os.path.dirname(__file__), "text4test.txt"), "r", encoding="utf-8"
) as f:
    for line in f:
        texts.append(line.rstrip("\n"))

target_words = ["apple"]
params = [("AdamW", 0.3, 5, texts, target_words)]


@pytest.mark.parametrize("optimizer, lr, n_itter, texts, target_words", params)
def test_hf_attack(optimizer, lr, n_itter, texts, target_words):
    n_sample = 10
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    target_ids = tokenizer.convert_tokens_to_ids(target_words)

    HFBA = HFBaseAttacker(
        victim_transformer=model,
        victim_tokenizer=tokenizer,
        optimizer=optimizer,
        init_val=15.0,
        lr=lr,
        grad_clip_max=10.0,
        grad_clip_norm=2.0,
        log_level=logging.DEBUG,
    )

    assert isinstance(HFBA, HFBaseAttacker)
    adv_tokens, log_coef = HFBA.attack_by_text(texts, target_ids, n_itter=n_itter)

    forbit_mask = HFBA.make_forbit_mask_from_texts(texts)

    sampled = HFBA.sample_texts(
        log_coef, n_sample=n_sample, original_texts=texts, forbid_mask=forbit_mask
    )
    assert isinstance(sampled, list)
    assert len(sampled) == n_sample
