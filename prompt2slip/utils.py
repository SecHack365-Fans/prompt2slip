from typing import Tuple, Optional, List, Union
from torchtyping import TensorType


def make_forbit_indicates(
    forbid_mask: Union[List[List[bool]], TensorType["batch", "length", bool]],
) -> List[List[int]]:
    forbid_indicates = []
    bsize = len(forbid_mask)
    for i in range(bsize):
        len_ = len(forbid_mask[i])
        forbid_indicates.append(
            [j for j in range(len_) if forbid_mask[i][j] == 1 or j < 2]
        )
    return forbid_indicates
