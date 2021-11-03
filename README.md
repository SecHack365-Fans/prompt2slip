# prompt2slip

![pCDlAAEvOFMmFk11635877148_1635877207](https://user-images.githubusercontent.com/32987034/139922987-55bce5fb-6dfd-476f-aebe-3daf661cdb94.png)


[![pytest](https://github.com/SecHack365-Fans/prompt2slip_proto/actions/workflows/pytest.yml/badge.svg)](https://github.com/SecHack365-Fans/prompt2slip_proto/actions/workflows/pytest.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/SecHack365-Fans/prompt2slip_proto/blob/main/LICENSE)

about this repos



## Install

````bash
pip install prompt2slip
````


## Usage

The simplest sample looks like this.


```python
import prompt2slip
from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

base_text = ["This project respects pytorch developers."]
target_word = ["keras"]

target_ids  = torch.Tensor(tokenizer.convert_tokens_to_ids(target_words))
attaker = CLMAttacker(model,tokenizer)
output = attacker.attack_by_text(texts,target_ids)
```

More realistic use cases are stored in [examples](https://github.com/SecHack365-Fans/prompt2slip_proto/tree/main/examples).


## References

- [Gradient-based Adversarial Attacks against Text Transformers ( Chuan Guo et al. ) ](https://arxiv.org/abs/2104.13733)

## License

[MIT License](https://github.com/SecHack365-Fans/prompt2slip_proto/blob/main/LICENSE)

## Development

### Install Package

Install [poetry](https://python-poetry.org/docs/#installation)

```bash
poetry install
```

### Test

Running tests with pytest.

```bash
poetry run pytest --cov .
```

