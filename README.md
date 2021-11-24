# prompt2slip
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->


![prompt2slip Logo](https://user-images.githubusercontent.com/32987034/140047469-32909981-eec5-4cfd-87ab-76010305b67f.png)


[![pytest](https://github.com/SecHack365-Fans/prompt2slip_proto/actions/workflows/pytest.yml/badge.svg)](https://github.com/SecHack365-Fans/prompt2slip/actions/workflows/pytest.yml)
[![release](https://github.com/SecHack365-Fans/prompt2slip/actions/workflows/release.yml/badge.svg?branch=main)](https://github.com/SecHack365-Fans/prompt2slip/actions/workflows/release.yml)
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

## What It Dose

"prompt2slip" provides the function to search for prompts which cause appearance of any specific word against a pre trained natural language generation model. Furthermore, with user customization, it can be applied to a wide range of tasks, including classification tasks.If you want to generate a hostile sample for a classification model, you can simply override the method to compute the adversarial loss function to generate a natural adversarial text.

The unique feature of this library is that it can generate test cases for verifying the danger of a pre-trained natural language model with a few lines of code.

## References

- [Gradient-based Adversarial Attacks against Text Transformers ( Chuan Guo et al. ) ](https://arxiv.org/abs/2104.13733)

## More About This Project

- [promp2slip | Devpost](https://devpost.com/software/promp2slip)

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


## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://www.kajyuuen.com/about"><img src="https://avatars.githubusercontent.com/u/15792784?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Koga Kobayashi</b></sub></a><br /><a href="https://github.com/SecHack365-Fans/prompt2slip/issues?q=author%3Akajyuuen" title="Bug reports">🐛</a> <a href="https://github.com/SecHack365-Fans/prompt2slip/commits?author=kajyuuen" title="Code">💻</a> <a href="https://github.com/SecHack365-Fans/prompt2slip/commits?author=kajyuuen" title="Documentation">📖</a> <a href="#example-kajyuuen" title="Examples">💡</a> <a href="#ideas-kajyuuen" title="Ideas, Planning, & Feedback">🤔</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!