from typing import Optional, Dict, Any, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from torchtyping import TensorType
import transformers

from .utils import make_forbit_indicates

KeyValBatchType = Union[transformers.BatchEncoding, Dict[str, Tensor]]


class CosSimLoss(nn.Module):
    def __init__(
        self, original_output: Optional[transformers.file_utils.ModelOutput]
    ) -> None:
        super(CosSimLoss, self).__init__()
        original_hidden: Optional(TensorType["batch", "length", "dim_hidden"]) = None
        if original_output is not None:
            self.original_hidden = original_output.hidden_states[0]

    def set_original(self, original_output: transformers.file_utils.ModelOutput):
        self.original_hidden = original_output.hidden_states[0]

    def forward(self, hf_output: transformers.file_utils.ModelOutput):
        if isinstance(self.original_hidden, torch.Tensor):
            expanded_bsize = hf_output.hidden_states[0].size(0)
            n_repeat = expanded_bsize // self.original_hidden.size(0)
            return (
                1
                - F.cosine_similarity(
                    self.original_hidden.repeat(n_repeat, 1, 1),
                    hf_output.hidden_states[0],
                    dim=2,
                )
            ).max()
        else:
            raise AttributeError("No `original_hidden`")


def log_perplexity(logits, coeffs):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_coeffs = coeffs[:, 1:, :].contiguous()
    shift_logits = shift_logits[:, :, : shift_coeffs.size(2)]
    return -(shift_coeffs * F.log_softmax(shift_logits, dim=-1)).sum(-1).mean()


class CrossEntropyLoss(nn.Module):
    def __init__(self, labels: Tensor) -> None:
        super(CrossEntropyLoss).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.labels = labels.long()

    def forward(self, hf_output: transformers.file_utils.ModelOutput):
        # hf_output.logits must be torch Tensor with [batch, n_labels]
        return self.loss_fn.forward(hf_output.logits, self.labels)


class MaxProbLoss(nn.Module):
    def __init__(
        self,
        margin: float = 1.5,
        forbid_mask: Optional[Tensor] = None,
        target_mask: Optional[Tensor] = None,
    ) -> None:
        super(MaxProbLoss, self).__init__()
        self.margin: float = margin

        if target_mask is None:
            self.target_mask = target_mask.bool()
            self.escape_mask = torch.logical_not(target_mask)
        else:
            self.target_mask = None
            self.escape_mask = None

    def forward(
        self,
        hf_output: transformers.file_utils.ModelOutput,
        target_ids: Tensor,
    ):
        assert isinstance(target_ids, Tensor)
        # hf_output.logits must be torch Tensor with [batch, length, n_vocabs]
        logits: TensorType["batch", "length", "n_vocabs"] = hf_output.logits

        target_ids_set = set(target_ids.tolist())

        self.n_vocab = logits.shape[2]
        non_target_ids = (
            torch.Tensor(
                [i for i in range(self.n_vocab) if i not in target_ids_set],
            )
            .long()
            .to(target_ids.device)
        )
        logits_log_softmax = F.log_softmax(logits, dim=2)
        target_logits: TensorType["batch", "length", "n_target"] = logits_log_softmax[
            :, :, target_ids
        ]
        nontarget_logits: TensorType[
            "batch", "length", "n_non_target"
        ] = logits_log_softmax[:, :, non_target_ids]
        max_tokens_target = torch.mean(target_logits, dim=2)  # (bsize, length)
        max_tokens_nontarget = torch.max(nontarget_logits, dim=2)[0]
        max_tokens_diff = torch.clamp(
            max_tokens_target - max_tokens_nontarget, max=self.margin
        )

        if isinstance(self.escape_mask, torch.Tensor):
            max_tokens_target = max_tokens_target.masked_fill_(self.escape_mask, -20)
            max_tokens_nontarget = max_tokens_nontarget.masked_fill_(
                self.escape_mask, -20
            )
            max_tokens_diff = max_tokens_diff.masked_fill_(self.escape_mask, -20)
        return (
            -1
            * torch.max(
                max_tokens_diff,
                dim=1,
            )[0].mean()
        )


def preprocess_embedding(
    embedding_vector: TensorType["batch", "length", "dim_embed"], **kargs
) -> Dict[str, Any]:
    kargs["inputs_embeds"] = embedding_vector
    kargs["output_hidden_states"] = True
    kargs["return_dict"] = True
    return kargs
