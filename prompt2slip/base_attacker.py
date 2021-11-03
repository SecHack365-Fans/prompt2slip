from typing import Tuple, Optional, Dict, List, Callable, Any, Union
from re import L

import logging
from logging import getLogger, StreamHandler, Formatter

import dataclasses

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from torchtyping import TensorType


from .calc_loss import EmbeddingSampler
from .utils import make_forbit_indicates


class BaseAttacker(object):
    def __init__(
        self,
        victim_encoder: nn.Module,
        victim_embedding: TensorType["vocabulary", "embedding"],
        refer_encoder: Optional[nn.Module] = None,
        refer_embedding: Optional[TensorType["vocabulary", "embedding"]] = None,
        optimizer: str = "Adam",
        init_val: float = 15.0,
        lr: float = 0.3,
        grad_clip_max: float = 10.0,
        grad_clip_norm: float = 2.0,
        weight_cossim: float = 1.0,
        weight_fluency: float = 1.0,
        max_batch_size: int = 16,
        log_level: int = logging.WARNING,
        device: str = "cpu",
    ) -> None:
        super(BaseAttacker, self).__init__()
        self.device = device
        self.embedding: Tensor = victim_embedding
        self.encoder: nn.Module = victim_encoder
        self.refer_embedding = (
            refer_embedding if refer_embedding is not None else victim_embedding
        )
        self.refer_encoder = (
            refer_encoder if refer_encoder is not None else victim_encoder
        )

        self.embedding.requires_grad = False
        self.refer_embedding.requires_grad = False
        self.optimzer_str: str = optimizer
        self.n_vocab: int = self.embedding.size(0)
        self.dim_embed = self.embedding.shape[1]
        self.init_val = init_val
        self.lr = lr
        self.weight_cossim = weight_cossim
        self.weight_fluency = weight_fluency
        self.max_batch_size = max_batch_size
        self.grad_clip_max = grad_clip_max
        self.grad_clip_norm = grad_clip_norm
        self.embedding_sampler_victim = EmbeddingSampler(self.embedding)
        self._embedding_layer = nn.Embedding(
            num_embeddings=self.n_vocab, embedding_dim=self.dim_embed
        )
        if refer_embedding is None:
            self.embedding_sampler_refer = self.embedding_sampler_victim
        else:
            self.embedding_sampler_refer = EmbeddingSampler(self.refer_embedding)
        self.logger = self._initialize_logger(log_level=log_level)

    def gumbel_softmax_and_bag_to_batch(
        self, log_coef: TensorType["batch", "length", "num_vocab"]
    ) -> Tuple[TensorType["batch_x_n_repeat", "length", "num_vocab"], int]:
        bsize = max(log_coef.size(0), 1)
        num_iter = self.max_batch_size // bsize
        coeffs_list = []
        for i in range(max(1, num_iter)):
            coeffs: TensorType["batch", "length", "num_vocab"] = F.gumbel_softmax(
                log_coef, hard=False
            )
            coeffs_list.append(coeffs)
        if len(coeffs_list) > 1:
            return torch.cat(coeffs_list, dim=0), num_iter
        else:
            return coeffs_list[0], 1

    def _initialize_logger(self, log_level) -> logging.RootLogger:
        logger = logging.getLogger()
        logger.setLevel(log_level)
        stream_handler = StreamHandler()
        stream_handler.setLevel(log_level)
        handler_format = Formatter(
            " [%(levelname)s] %(asctime)s | %(module)s.%(funcName)s.%(lineno)d : %(message)s"
        )
        stream_handler.setFormatter(handler_format)
        logger.addHandler(stream_handler)
        return logger

    def _get_victim_hidden_from_ids(
        self, original_ids: TensorType["batch", "length", int], **kargs
    ):
        original_embedding = self._embedding_layer(original_ids)
        return self.encoder.forward(original_embedding)

    def _initialize_loss_func(
        self,
        original_ids: Optional[TensorType["batch", "length", int]],
        target_tensor: Tensor,
        forbid_mask: Optional[TensorType["batch", "length", bool]] = None,
        **kargs,
    ) -> None:
        pass

    def _calc_loss(
        self,
        log_coef: TensorType["batch", "length", "num_vocab"],
        target_tensor: Tensor,
        **kargs,
    ) -> Tensor:
        # some calculation
        # This is dummy process to pass the test case.
        # You have to customize this method.
        out = log_coef.max() + target_tensor.max()
        return out

    def attack(
        self,
        original_ids: TensorType["batch", "length", int],
        target_tensor: Tensor,
        forbid_mask: Optional[TensorType["batch", "length", bool]] = None,
        n_itter: int = 40,
        n_sample: int = 10,
        **kargs,
    ) -> Tuple[List[Tensor], Tensor]:

        original_ids = original_ids.to(self.device)
        target_tensor = target_tensor.to(self.device)
        if isinstance(forbid_mask, torch.Tensor):
            forbit = forbid_mask.to(self.device)
        for k in kargs.keys():
            if isinstance(kargs, torch.Tensor):
                kargs[k] = kargs[k].to(self.device)
        self.logger.info("Start Attack")
        self.encoder.eval()
        length = original_ids.size(1)
        bsize = original_ids.shape[0]
        forbit_indicates = make_forbit_indicates(forbid_mask)
        log_coef: TensorType["batch", "length", "num_vocab"] = torch.zeros(
            size=(bsize, length, self.n_vocab),
            device=original_ids.device,
        ).to(self.device)
        indicates = torch.arange(0, length).long().to(self.device)
        for i in range(bsize):
            log_coef[i, indicates, original_ids[i].long()] = self.init_val
        for i in range(bsize):
            log_coef[
                i,
                indicates[[forbit_indicates[i]]],
                original_ids[i][forbit_indicates[i]].long(),
            ] = (
                self.init_val * 10
            )
        log_coef.requires_grad_(True)
        optimizer = eval(f"torch.optim.{self.optimzer_str}")([log_coef], lr=self.lr)

        self.encoder.eval()
        self.refer_encoder.eval()
        self.encoder.zero_grad()
        self.refer_encoder.zero_grad()
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.refer_encoder.parameters():
            param.requires_grad = False
        self.encoder.to(self.device)
        self.refer_encoder.to(self.device)
        self.embedding.to(self.device)
        self.refer_encoder.to(self.device)

        # initialize loss function
        self._initialize_loss_func(original_ids, target_tensor, forbid_mask, **kargs)

        for i in range(n_itter):
            optimizer.zero_grad()
            loss2opt = self._calc_loss(log_coef, target_tensor, **kargs)
            loss2opt.backward()
            torch.nn.utils.clip_grad_norm_(
                [log_coef],
                self.grad_clip_max,
                norm_type=self.grad_clip_norm,
                error_if_nonfinite=False,
            )
            if isinstance(forbid_mask, torch.Tensor):
                log_coef.grad.masked_fill_(
                    forbid_mask.unsqueeze(2).repeat(1, 1, self.n_vocab), 0
                )
            optimizer.step()

        generated_ids = self.sample_ids(log_coef, n_sample=n_sample)

        generated_ids = self._replace_forbit_ids(
            generated_ids, forbid_mask, original_ids
        )

        return generated_ids, log_coef

    def _replace_forbit_ids(
        self,
        generated_ids: List[TensorType["batch", "length", torch.long]],
        forbid_mask: Union[List[List[bool]], TensorType["batch", "length", bool]],
        original_ids: TensorType["batch", "length", int],
    ) -> List[TensorType["batch", "length", torch.long]]:
        forbid_indicates = make_forbit_indicates(forbid_mask)
        bsize = len(forbid_mask)
        for h in range(len(generated_ids)):
            for i in range(bsize):
                generated_ids[h][i][forbid_indicates[i]] = original_ids[i][
                    forbid_indicates[i]
                ]
        return generated_ids

    def sample_ids(
        self,
        log_coef: TensorType["batch", "length", "num_vocab"],
        n_sample: int = 10,
    ) -> List[TensorType["batch", "length", int]]:
        generated_ids = []
        for i in range(n_sample):
            adv_ids = F.gumbel_softmax(log_coef, hard=True).argmax(2)
            generated_ids.append(adv_ids)
        return generated_ids
