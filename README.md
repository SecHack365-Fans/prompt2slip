# prompt2slip


![prompt2slip Logo](https://user-images.githubusercontent.com/32987034/140047469-32909981-eec5-4cfd-87ab-76010305b67f.png)


[![pytest](https://github.com/SecHack365-Fans/prompt2slip_proto/actions/workflows/pytest.yml/badge.svg)](https://github.com/SecHack365-Fans/prompt2slip/actions/workflows/pytest.yml)
[![PyPI version](https://badge.fury.io/py/prompt2slip.svg)](https://badge.fury.io/py/prompt2slip)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/SecHack365-Fans/prompt2slip/blob/main/LICENSE)

This library is testing the ethics of language models by using natural adversarial texts.

This tool allows for short and simple code and validation with little effort.

Extensibility to be applied to any language model in any problem setting by inheriting from the base class.



## Install

```bash
pip install prompt2slip
```


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

More realistic use cases are stored in [examples](https://github.com/SecHack365-Fans/prompt2slip/tree/main/examples).


## References

- [Gradient-based Adversarial Attacks against Text Transformers ( Chuan Guo et al. ) ](https://arxiv.org/abs/2104.13733)

## License

[MIT License](https://github.com/SecHack365-Fans/prompt2slip/blob/main/LICENSE)

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

