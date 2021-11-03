from typing import Tuple, Optional, List, Union

import logging
from numpy import not_equal

import torch
from torch import nn
from torch._C import ListType
from torch.nn import functional as F
from torch import Tensor
import transformers
from torchtyping import TensorType

from .base_attacker import BaseAttacker
from . import loss_transformers


def get_HF_inputs(kargs):
    keys = [
        "attention_mask",
        "token_type_ids",
        "position_ids",
        "head_mask",
        "inputs_embeds",
        "output_attentions",
        "output_hidden_states",
        "return_dict",
        "labels",
    ]
    new_kargs = {}
    for k in keys:
        v = kargs.get(k, None)
        if isinstance(v, torch.Tensor):
            new_kargs[k] = v
    return new_kargs


class HFBaseAttacker(BaseAttacker):
    def __init__(
        self,
        victim_transformer: transformers.PreTrainedModel,
        victim_tokenizer: transformers.PreTrainedTokenizer,
        refer_transformer: Optional[nn.Module] = None,
        refer_tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
        optimizer: str = "Adam",
        init_val: float = 15,
        lr: float = 0.3,
        grad_clip_max: float = 10,
        grad_clip_norm: float = 2,
        margin: float = 0.1,
        max_batch_size: int = 16,
        weight_cossim: float = 1.0,
        weight_fluency: float = 1.0,
        log_level: int = logging.WARNING,
    ) -> None:
        self.victim_tokenizer = victim_tokenizer
        if self.victim_tokenizer.pad_token is None:
            self.victim_tokenizer.pad_token = self.victim_tokenizer.eos_token
        victim_embedding = self.get_embedding_matrix(
            victim_transformer, victim_tokenizer.vocab_size
        ).detach()
        if refer_transformer is None:
            refer_embedding = None
        else:
            if refer_tokenizer is None:
                self.refer_tokenizer = self.victim_tokenizer
                refer_embedding = victim_embedding
            else:
                self.refer_tokenizer = refer_tokenizer
                if self.refer_tokenizer.pad_token is None:
                    self.refer_tokenizer.pad_token = self.refer_tokenizer.eos_token
                refer_embedding = self.get_embedding_matrix(
                    refer_transformer, refer_tokenizer.vocab_size
                ).detach()
        super(HFBaseAttacker, self).__init__(
            victim_transformer,
            victim_embedding,
            refer_encoder=refer_transformer,
            refer_embedding=refer_embedding,
            optimizer=optimizer,
            init_val=init_val,
            lr=lr,
            grad_clip_max=grad_clip_max,
            grad_clip_norm=grad_clip_norm,
            weight_cossim=weight_cossim,
            weight_fluency=weight_fluency,
            log_level=log_level,
            max_batch_size=max_batch_size,
        )
        self.adv_loss_fn = lambda x: None
        self.margin = margin
        if refer_transformer is None:
            assert id(self.refer_encoder) == id(self.encoder)

        tokens_to_fobit = []
        for s in [
            "bos",
            "eos",
            "unk",
            "sep",
            "pad",
            "cls",
            "mask",
            "additional_special",
        ]:
            try:
                t = eval(f"self.victim_tokenizer.{s}_token_id")
                tokens_to_fobit.append(t)
            except:
                pass
        self.tokens_to_forbit = set(tokens_to_fobit)

    def make_forbit_mask_from_input_ids(
        self, input_ids: List[List[int]]
    ) -> List[List[bool]]:
        forbid_mask = []
        for ids in input_ids:
            # print(self.victim_tokenizer.convert_ids_to_tokens(ids))
            forbit_mask_by_text = self.victim_tokenizer.get_special_tokens_mask(ids)
            forbit_mask_by_text = list(map(bool, forbit_mask_by_text))
            for i, id_ in enumerate(ids):
                if id_ in self.tokens_to_forbit:
                    forbit_mask_by_text[i] = True
            forbid_mask.append(forbit_mask_by_text)
        return forbid_mask

    def make_forbit_mask_from_texts(self, texts: List[str]) -> List[List[int]]:
        input_batch = self.victim_tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        input_ids = input_batch.pop("input_ids")
        return self.make_forbit_mask_from_input_ids(input_ids=input_ids.tolist())

    def get_embedding_matrix(
        cls, model: transformers.PreTrainedModel, vocab_size: int
    ) -> TensorType["num_vocab", "dim_embedding"]:
        return model.get_input_embeddings()(torch.arange(0, vocab_size).long())

    def _get_victim_hidden_from_ids(
        self, original_ids: TensorType["batch", "length", int], **kargs
    ):
        return self.encoder.forward(input_ids=original_ids, **kargs)

    def initialize_adv_loss(
        self,
        original_ids: Optional[TensorType["batch", "length", int]],
        target_tensor: Tensor,
        forbid_mask: Optional[TensorType["batch", "length", bool]] = None,
        **kargs,
    ) -> None:
        pass

    def _get_adv_loss(
        self,
        output_of_transformer: transformers.file_utils.ModelOutput,
        target_tensor: Tensor,
    ) -> Tensor:
        # output_of_transformer.logits :[batch, length, logits]
        return (
            F.softmax(output_of_transformer.logits, dim=-1)[:, :, target_tensor.long()]
            .max(dim=-1)[0]
            .mean()
        )

    def _initialize_loss_func(
        self,
        original_ids: Optional[TensorType["batch", "length", int]],
        target_tensor: Tensor,
        forbid_mask: Optional[TensorType["batch", "length", bool]] = None,
        **kargs,
    ) -> None:
        # get original embedding for similarity constraint.
        original_output = self.refer_encoder(
            input_ids=original_ids,
            return_dict=True,
            output_hidden_states=True,
            **get_HF_inputs(kargs),
        )
        self.cossim_nn = loss_transformers.CosSimLoss(original_output)
        self.cossim_nn.zero_grad()
        for param in self.cossim_nn.parameters():
            param.requires_grad = False
        self.initialize_adv_loss(
            original_ids,
            target_tensor,
            forbid_mask,
            **kargs,
        )

    def _calc_loss(
        self,
        log_coef: TensorType["batch", "length", "num_vocab"],
        target_tensor: Tensor,
        **kargs,
    ) -> Tensor:
        # some calculation
        # This is dummy process to pass the test case.
        # You have to customize this method.
        coeffs, n_repeat = self.gumbel_softmax_and_bag_to_batch(log_coef)
        # coeffs = F.gumbel_softmax(log_coef, hard=False)
        # repeat kargs
        for k in kargs.keys():
            if isinstance(kargs[k], torch.Tensor):
                shape_of_k = kargs[k].size()
                repeat_list = [1] * len(shape_of_k)
                repeat_list[0] = n_repeat
                kargs[k] = kargs[k].repeat(*repeat_list)

        # input_embeds ["batch", "length", "dim_embed"]
        input_embeds = coeffs @ self.embedding.detach()
        input_ref_embeds = coeffs @ self.refer_embedding.detach()
        out_victim = self.encoder(
            inputs_embeds=input_embeds,
            **get_HF_inputs(kargs),
            return_dict=True,
            output_hidden_states=True,
        )
        # リファレンス用のエンコーダーがvictimのエンコーダーと同じものであれば、二重に勾配を伝搬させないようにする
        if id(self.refer_encoder) == id(self.encoder):
            out_ref = out_victim
        else:
            out_ref = self.refer_encoder(
                inputs_embeds=input_ref_embeds,
                **get_HF_inputs(kargs),
                return_dict=True,
                output_hidden_states=True,
            )
        loss_cossim = self.cossim_nn(out_ref) * self.weight_cossim
        loss_fruently = (
            loss_transformers.log_perplexity(out_ref.logits, coeffs)
            * self.weight_fluency
        )
        loss_adv = self._get_adv_loss(out_victim, target_tensor)
        self.logger.debug(
            f"{loss_adv.detach().item():.4}, {loss_cossim.detach().item()/self.weight_cossim:.4}, {loss_fruently.detach().item()/self.weight_fluency:.4}"
        )
        return loss_cossim + loss_fruently + loss_adv

    def attack_by_text(
        self,
        texts: List[str],
        target: Union[Tensor, List[int], int],
        forbid_mask: Optional[
            Union[TensorType["batch", "length", bool], List[List[bool]]]
        ] = None,
        n_itter: int = 40,
        n_sample: int = 10,
        **kargs,
    ) -> Tuple[List[List[List[str]]], Tensor]:

        input_batch = self.victim_tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        input_ids = input_batch.pop("input_ids")
        # print("input ids 167", input_ids)
        if isinstance(target, list):
            target = torch.Tensor(target).long()
        if forbid_mask is None:
            forbid_mask = self.make_forbit_mask_from_input_ids(input_ids.tolist())
            # print("forbid_mask 170: ", forbid_mask)
        else:
            if isinstance(forbid_mask, list):
                forbid_mask = torch.Tensor(forbid_mask, dtype=torch.bool)

        for k, v in input_batch.items():
            kargs[k] = v
        id_list, log_coef = super(HFBaseAttacker, self).attack(
            input_ids,
            target,
            forbid_mask=forbid_mask,
            n_itter=n_itter,
            n_sample=n_sample,
            **kargs,
        )
        list_of_tokens = self.id_list_to_token_list(id_list)

        return list_of_tokens, log_coef

    def sample_texts(
        self,
        log_coef: TensorType["batch", "length", "num_vocab"],
        n_sample: int = 10,
        original_texts: Optional[List[str]] = None,
        forbid_mask: Optional[
            Union[List[List[bool]], TensorType["batch", "length", bool]]
        ] = None,
    ) -> List[TensorType["batch", "length", torch.long]]:
        id_list = super(HFBaseAttacker, self).sample_ids(
            log_coef=log_coef, n_sample=n_sample
        )
        if original_texts is not None:
            original_batch = self.victim_tokenizer(
                original_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            original_ids = original_batch.pop("input_ids")
            if forbid_mask is None:
                forbid_mask = self._make_forbit_mask_from_input_ids(
                    original_ids.tolist()
                )
            id_list = self._replace_forbit_ids(id_list, forbid_mask, original_ids)
        return self.id_list_to_token_list(id_list=id_list)

    def id_list_to_token_list(
        self, id_list: List[TensorType["batch", "length", int]]
    ) -> List[List[List[str]]]:
        list_of_tokens = [
            [
                self.victim_tokenizer.convert_ids_to_tokens(ids_sentence.tolist())
                for ids_sentence in ids_batch.detach().cpu()
            ]
            for ids_batch in id_list
        ]
        return list_of_tokens


class CLMAttacker(HFBaseAttacker):
    def __init__(
        self,
        victim_transformer: transformers.PreTrainedModel,
        victim_tokenizer: transformers.PreTrainedTokenizer,
        refer_transformer: Optional[nn.Module] = None,
        refer_tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
        optimizer: str = "Adam",
        init_val: float = 15,
        lr: float = 0.3,
        grad_clip_max: float = 10,
        grad_clip_norm: float = 2,
        margin: float = 0.1,
        max_batch_size: int = 16,
        weight_cossim: float = 1,
        weight_fluency: float = 1,
        log_level: int = logging.WARNING,
    ) -> None:
        super(CLMAttacker, self).__init__(
            victim_transformer,
            victim_tokenizer,
            refer_transformer,
            refer_tokenizer,
            optimizer=optimizer,
            init_val=init_val,
            lr=lr,
            grad_clip_max=grad_clip_max,
            grad_clip_norm=grad_clip_norm,
            margin=margin,
            weight_cossim=weight_cossim,
            weight_fluency=weight_fluency,
            log_level=log_level,
            max_batch_size=max_batch_size,
        )

    def initialize_adv_loss(
        self,
        original_ids: Optional[TensorType["batch", "length", int]],
        target_tensor: Tensor,
        forbid_mask: Optional[TensorType["batch", "length", bool]] = None,
        **kargs,
    ) -> None:
        target_mask = kargs.get("target_mask", None)
        self.adv_loss_fn = loss_transformers.MaxProbLoss(
            margin=self.margin, forbid_mask=forbid_mask, target_mask=True
        )
        for param in self.adv_loss_fn.parameters():
            param.requires_grad = False

    def _get_adv_loss(
        self,
        output_of_transformer: transformers.file_utils.ModelOutput,
        target_tensor: Tensor,
    ) -> Tensor:
        self.adv_loss_fn.zero_grad()
        return self.adv_loss_fn(output_of_transformer, target_tensor)


class ClassificationAttacker(HFBaseAttacker):
    def __init__(
        self,
        victim_transformer: transformers.PreTrainedModel,
        victim_tokenizer: transformers.PreTrainedTokenizer,
        refer_transformer: Optional[nn.Module] = None,
        refer_tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
        optimizer: str = "Adam",
        init_val: float = 15,
        lr: float = 0.3,
        grad_clip_max: float = 10,
        grad_clip_norm: float = 2,
        margin: float = 0.1,
        max_batch_size: int = 16,
        weight_cossim: float = 1,
        weight_fluency: float = 1,
        log_level: int = logging.WARNING,
    ) -> None:
        super(CLMAttacker, self).__init__(
            victim_transformer,
            victim_tokenizer,
            refer_transformer,
            refer_tokenizer,
            optimizer=optimizer,
            init_val=init_val,
            lr=lr,
            grad_clip_max=grad_clip_max,
            grad_clip_norm=grad_clip_norm,
            margin=margin,
            weight_cossim=weight_cossim,
            weight_fluency=weight_fluency,
            log_level=log_level,
            max_batch_size=max_batch_size,
        )

    def initialize_adv_loss(
        self,
        original_ids: Optional[TensorType["batch", "length", int]],
        target_tensor: Tensor,
        forbid_mask: Optional[TensorType["batch", "length", bool]] = None,
        **kargs,
    ) -> None:
        target_mask = kargs.get("target_mask", None)
        self.adv_loss_fn = loss_transformers.MaxProbLoss(
            margin=self.margin, forbid_mask=forbid_mask, target_mask=True
        )
        for param in self.adv_loss_fn.parameters():
            param.requires_grad = False

    def _get_adv_loss(
        self,
        output_of_transformer: transformers.file_utils.ModelOutput,
        target_tensor: Tensor,
    ) -> Tensor:
        self.adv_loss_fn.zero_grad()
        return self.adv_loss_fn(output_of_transformer, target_tensor)
