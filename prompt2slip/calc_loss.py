from typing import Optional, Dict, List, Any, Union, Callable


from torch import nn
from torch.nn import functional as F
from torch import Tensor
from torchtyping import TensorType
import transformers


KeyValBatchType = Union[transformers.BatchEncoding, Dict[str, Tensor]]


class EmbeddingSampler(nn.Module):
    def __init__(self, embedding: TensorType["vocabulary", "embedding"]) -> None:
        super(EmbeddingSampler, self).__init__()
        self.embedding_matrix = embedding

    def forward(
        self, log_coef: TensorType["batch", "length", "num_vocab"]
    ) -> TensorType["batch", "length", "dim_embed"]:
        if self.training:
            hard = False
        else:
            hard = True
        coeffs = F.gumbel_softmax(log_coef, hard=hard)
        # coef: TensorType["batch", "length", "num_vocab"]
        # input_embeds ["batch", "length", "dim_embed"]
        input_embeds = coeffs @ self.embedding_matrix.detach()
        return input_embeds


class AdvCrossEntropyLoss(nn.Module):
    def __init__(self, victim_encoder: nn.Module) -> None:
        super(AdvCrossEntropyLoss, self).__init__()
        self.victim_encoder = victim_encoder
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        embedding_vector: TensorType["batch", "length", "dim_embed"],
        target_tensor: TensorType["batch"],
    ) -> Tensor:
        # logits ["batch", "n_labels"]
        logits = self.victim_encoder.forward(embedding_vector)
        loss = self.loss_fn.forward(logits, target_tensor)
        return loss


class GeneralLMLoss(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        postproc_modules=List[nn.Module],
        weights_postproc=List[float],
        preproc_input: Optional[Callable[[Tensor, Any], KeyValBatchType]] = None,
        preproc_target: Optional[Callable[[Tensor, Any], KeyValBatchType]] = None,
    ) -> None:
        super(GeneralLMLoss).__init__()
        self.encoder = encoder
        self.postproc_module = nn.ModuleList(postproc_modules)
        self.weights_postproc = weights_postproc
        self.preproc_input = preproc_input
        self.preproc_target = preproc_target

    def forward(
        self,
        embedding_vector: TensorType["batch", "length", "dim_embed"],
        target_tensor: TensorType["batch"],
        **kargs,
    ) -> Tensor:

        if self.preproc_func is None:
            input_ = self.preproc_input(embedding_vector, **kargs)
        else:
            input_ = embedding_vector
        if self.preproc_target is None:
            target_ = self.preproc_target(target_tensor, **kargs)
        else:
            target_ = target_tensor
        if isinstance(input_, Tensor):
            output = self.victim_encoder.forward(input_)
        else:
            output = self.victim_encoder.forward(**input_)
        # logits ["batch", "n_labels"]
        loss = 0
        for i in range(self.postproc_module):
            loss += self.postproc_module[i].forward(output) + self.weights_postproc[i]
        return loss
