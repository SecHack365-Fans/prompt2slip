# prompt2slip


![prompt2slip Logo](https://user-images.githubusercontent.com/32987034/140047469-32909981-eec5-4cfd-87ab-76010305b67f.png)


[![pytest](https://github.com/SecHack365-Fans/prompt2slip_proto/actions/workflows/pytest.yml/badge.svg)](https://github.com/SecHack365-Fans/prompt2slip/actions/workflows/pytest.yml)
[![release](https://github.com/SecHack365-Fans/prompt2slip/actions/workflows/release.yml/badge.svg?branch=main)](https://github.com/SecHack365-Fans/prompt2slip/actions/workflows/release.yml)
[![PyPI version](https://badge.fury.io/py/prompt2slip.svg)](https://badge.fury.io/py/prompt2slip)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/SecHack365-Fans/prompt2slip/blob/main/LICENSE)
[![Contributors](https://img.shields.io/badge/all_contributors-4-orange.svg)](#contributors)

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
from prompt2slip import CLMAttacker
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

base_text = ["This project respects pytorch developers."]
target_words = ["keras"]

target_ids = torch.Tensor(tokenizer.convert_tokens_to_ids(target_words)).long()
attacker = CLMAttacker(model, tokenizer)
output = attacker.attack_by_text(base_text, target_ids)
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


## Contributors

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/akiFQC"><img src="https://avatars.githubusercontent.com/u/32811500?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Aki Fukuchi</b></sub></a><br /><a href="https://github.com/SecHack365-Fans/prompt2slip/commits?author=akiFQC" title="Code">💻</a> <a href="https://github.com/SecHack365-Fans/prompt2slip/issues?q=author%3AakiFQC" title="Bug reports">🐛</a> <a href="#content-akiFQC" title="Content">🖋</a> <a href="#example-akiFQC" title="Examples">💡</a> <a href="#video-akiFQC" title="Videos">📹</a> <a href="https://github.com/SecHack365-Fans/prompt2slip/pulls?q=is%3Apr+reviewed-by%3AakiFQC" title="Reviewed Pull Requests">👀</a> <a href="#mentoring-akiFQC" title="Mentoring">🧑‍🏫</a></td>
    <td align="center"><a href="http://task4233.dev"><img src="https://avatars.githubusercontent.com/u/29667656?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Takashi MIMA</b></sub></a><br /><a href="https://github.com/SecHack365-Fans/prompt2slip/issues?q=author%3Atask4233" title="Bug reports">🐛</a> <a href="https://github.com/SecHack365-Fans/prompt2slip/commits?author=task4233" title="Code">💻</a> <a href="https://github.com/SecHack365-Fans/prompt2slip/commits?author=task4233" title="Documentation">📖</a> <a href="https://github.com/SecHack365-Fans/prompt2slip/commits?author=task4233" title="Tests">⚠️</a> <a href="#video-task4233" title="Videos">📹</a></td>
    <td align="center"><a href="https://www.kajyuuen.com/about"><img src="https://avatars.githubusercontent.com/u/15792784?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Koga Kobayashi</b></sub></a><br /><a href="#example-kajyuuen" title="Examples">💡</a> <a href="#ideas-kajyuuen" title="Ideas, Planning, & Feedback">🤔</a> <a href="#question-kajyuuen" title="Answering Questions">💬</a> <a href="https://github.com/SecHack365-Fans/prompt2slip/pulls?q=is%3Apr+reviewed-by%3Akajyuuen" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/SecHack365-Fans/prompt2slip/commits?author=kajyuuen" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://github.com/Uno-Takashi"><img src="https://avatars.githubusercontent.com/u/32987034?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Uno-Takashi</b></sub></a><br /><a href="https://github.com/SecHack365-Fans/prompt2slip/issues?q=author%3AUno-Takashi" title="Bug reports">🐛</a> <a href="https://github.com/SecHack365-Fans/prompt2slip/commits?author=Uno-Takashi" title="Code">💻</a> <a href="https://github.com/SecHack365-Fans/prompt2slip/commits?author=Uno-Takashi" title="Documentation">📖</a> <a href="#ideas-Uno-Takashi" title="Ideas, Planning, & Feedback">🤔</a> <a href="#platform-Uno-Takashi" title="Packaging/porting to new platform">📦</a> <a href="https://github.com/SecHack365-Fans/prompt2slip/pulls?q=is%3Apr+reviewed-by%3AUno-Takashi" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/SecHack365-Fans/prompt2slip/commits?author=Uno-Takashi" title="Tests">⚠️</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
