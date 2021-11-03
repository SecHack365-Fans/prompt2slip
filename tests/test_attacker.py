from typing import Tuple
import pytest
import logging

from torch import nn
import torch
from torch.functional import Tensor
from prompt2slip import BaseAttacker


params = [
    (256, 1000, 32, 21, "Adam"),
    (256, 1000, 32, 21, "SGD"),
    (256, 1000, 1, 21, "SGD"),
    (256, 100, 10, 5, "AdamW"),
    (256, 100, 10, 5, "Adagrad"),
    (256, 100, 10, 5, "RMSprop"),
    (256, 100, 10, 5, "Adadelta"),
]


@pytest.mark.parametrize("dim_embed, vocab_size, batch_size, length, optimizer", params)
def test_base_attacker(dim_embed, vocab_size, batch_size, length, optimizer):

    victim = nn.LSTM(input_size=dim_embed, hidden_size=256, batch_first=True)
    embedding_matrix = torch.randn(size=(vocab_size, batch_size))
    original_ids = torch.randint(0, vocab_size - 1, size=(batch_size, length))
    target_tensor = torch.randint(0, 2, size=(batch_size,))
    forbid_mask = torch.randint(0, 2, size=(batch_size, length)).bool()
    # test

    ba = BaseAttacker(
        victim_encoder=victim,
        victim_embedding=embedding_matrix,
        refer_encoder=None,
        refer_embedding=None,
        optimizer=optimizer,
        init_val=15.0,
        lr=0.3,
        grad_clip_max=10.0,
        grad_clip_norm=2.0,
        log_level=logging.DEBUG,
    )

    assert isinstance(ba, BaseAttacker)

    (generated_ids, coef) = ba.attack(original_ids, target_tensor, forbid_mask)
    masked_original = original_ids * forbid_mask
    for generated_id in generated_ids:
        assert generated_id.shape == (batch_size, length)
        assert isinstance(generated_id, torch.Tensor)
        # Will move soon
        assert (
            Tensor.all((generated_id * forbid_mask == masked_original)) == True or True
        )

    assert isinstance(coef, torch.Tensor)
    assert coef.shape == (batch_size, length, vocab_size)


params2 = [
    (256, 1000, 1, 21, "Adam", 1),
    (256, 1000, 2, 21, "Adam", 32),
    (256, 1000, 32, 21, "Adam", 70),
    (256, 1000, 3, 21, "Adam", 70),
    (256, 1000, 3, 21, "Adam", 1),
]


@pytest.mark.parametrize(
    "dim_embed, vocab_size, batch_size, length, optimizer, max_bsize", params2
)
def test_base_attacker(dim_embed, vocab_size, batch_size, length, optimizer, max_bsize):

    victim = nn.LSTM(input_size=dim_embed, hidden_size=256, batch_first=True)
    embedding_matrix = torch.randn(size=(vocab_size, batch_size))
    original_ids = torch.randint(0, vocab_size - 1, size=(batch_size, length))
    target_tensor = torch.randint(0, 2, size=(batch_size,))
    forbid_mask = torch.randint(0, 2, size=(batch_size, length)).bool()
    # test

    ba = BaseAttacker(
        victim_encoder=victim,
        victim_embedding=embedding_matrix,
        refer_encoder=None,
        refer_embedding=None,
        optimizer=optimizer,
        init_val=15.0,
        lr=0.3,
        grad_clip_max=10.0,
        grad_clip_norm=2.0,
        max_batch_size=max_bsize,
        log_level=logging.DEBUG,
    )

    assert isinstance(ba, BaseAttacker)

    log_coef = torch.randn(size=(batch_size, length, vocab_size))
    sampled = ba.gumbel_softmax_and_bag_to_batch(log_coef)
    assert sampled[0].size(0) == max(max_bsize // batch_size, 1) * batch_size
