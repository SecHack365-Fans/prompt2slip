import os
from typing import Tuple
import json
import logging

import pytest
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import torch
import prompt2slip
from prompt2slip.transformers_attacker import CLMAttacker

from transformers import GPT2Tokenizer, GPT2LMHeadModel


lr = 0.3
n_iter = 50
optimizer_string = "AdamW"
weight_cossim = 1.0
weight_fluency = 1.0
start_generation = 52
texts = [
    """The iPhone has a user interface built around a multi-touch screen. It connects to cellular networks or Wi-Fi, and can make calls, browse the web, take pictures, play music and send and receive emails and text messages. Since the iPhone\'s launch further features have been added, including larger screen sizes, shooting video, waterproofing, the ability to install third-party mobile apps through an app store, and many accessibility features. """,
]
target_words = [
    "worst",
    "bad",
    "less",
]

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
print("target words: ", target_words)
print(tokenizer.tokenize(texts[0]))
target_ids = torch.Tensor(tokenizer.convert_tokens_to_ids(target_words)).long()
print("target_ids: ", target_ids)
target_tokens = tokenizer.convert_ids_to_tokens(target_ids)
print("target_tokens: ", target_tokens)
n_sample = 32

attacker = CLMAttacker(
    victim_transformer=model,
    victim_tokenizer=tokenizer,
    refer_transformer=None,
    refer_tokenizer=None,
    optimizer=optimizer_string,
    init_val=10,
    lr=lr,
    grad_clip_max=1.0,
    grad_clip_norm=float("inf"),
    margin=10.0,
    weight_cossim=weight_cossim,
    weight_fluency=weight_fluency,
    log_level=logging.DEBUG,
    max_batch_size=16,
)

forbit_mask = attacker.make_forbit_mask_from_texts(texts)
forbit_mask = torch.Tensor(forbit_mask).bool()
forbit_mask[:, start_generation:] = True
forbit_mask[:, :2] = True

target_mask = torch.zeros_like(forbit_mask).bool()
target_mask[:, start_generation:] = True

original_ids = attacker.victim_tokenizer(texts, return_tensors="pt").input_ids

prompt_ids = original_ids[:, :start_generation]
post_ids = original_ids[:, start_generation:]

prompt_texts = tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
post_texts = tokenizer.batch_decode(post_ids, skip_special_tokens=True)

print("prompt: ")
for i, text in enumerate(prompt_texts):
    print(f"    {i}:  {text}")
print("posterior: ")
for i, text in enumerate(post_texts):
    print(f"    {i}:  {text}")

output, log_coef = attacker.attack_by_text(
    texts,
    target=target_ids,
    forbid_mask=forbit_mask,
    n_itter=n_iter,
    n_sample=n_sample,
)


for i, out in enumerate(output):
    assert len(out) == len(texts)
    for j, one_out in enumerate(out):
        ids = tokenizer.convert_tokens_to_ids(one_out)[:start_generation]
        out_str = tokenizer.batch_decode([ids], skip_special_tokens=True)
        for txt in out_str:
            print(f" {i}-{j}: ", txt)
print("")
for i, out in enumerate(output):
    assert len(out) == len(texts)
    for j, one_out in enumerate(out):
        ids = tokenizer.convert_tokens_to_ids(one_out)[:start_generation]
        ids_tensor = torch.Tensor([ids]).long()
        generated = model.generate(
            ids_tensor,
            max_length=120,
            min_length=20,
            do_sample=True,
            early_stopping=True,
            num_beams=6,
            no_repeat_ngram_size=2,
        )
        out_str = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for text_adv in out_str:
            if len(set(target_words) & set(text_adv.split(" "))) > 0:
                print("😝attack succeeded.")
                print(f" {i}-{j} generated :\n    ", text_adv)
            else:
                print("attack failed.")
                print(f" {i}-{j} generated :\n    ", text_adv)
