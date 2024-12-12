URL: https://huggingface.co/mistralai/Mistral-7B-v0.1
---
## You need to agree to share your contact information to access this model

If you want to learn more about how we process your personal data, please read our [Privacy Policy](https://mistral.ai/terms/).

[Log in](/login?next=%2Fmistralai%2FMistral-7B-v0.1)
or
[Sign Up](/join?next=%2Fmistralai%2FMistral-7B-v0.1)
to review the conditions and access this model content.

# Model Card for Mistral-7B-v0.1

The Mistral-7B-v0.1 Large Language Model (LLM) is a pretrained generative text model with 7 billion parameters.
Mistral-7B-v0.1 outperforms Llama 2 13B on all benchmarks we tested.

For full details of this model please read our [paper](https://arxiv.org/abs/2310.06825) and [release blog post](https://mistral.ai/news/announcing-mistral-7b/).

## Model Architecture

Mistral-7B-v0.1 is a transformer model, with the following architecture choices:

- Grouped-Query Attention
- Sliding-Window Attention
- Byte-fallback BPE tokenizer

## Troubleshooting

- If you see the following error:

```
KeyError: 'mistral'

```

- Or:

```
NotImplementedError: Cannot copy out of meta tensor; no data!

```

Ensure you are utilizing a stable version of Transformers, 4.34.0 or newer.

## Notice

Mistral 7B is a pretrained base model and therefore does not have any moderation mechanisms.

## The Mistral AI Team

Albert Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed.

Downloads last month886,907


Safetensors

Model size

7.24B params

Tensor type

BF16

·

Inference API

warm

[Text Generation](/tasks/text-generation "Learn more about text-generation")

Examples

My name is Julien and I like to make th

`ctrl+Enter`

View Code

0.3s

Maximize

## Model tree for mistralai/Mistral-7B-v0.1

Adapters

[1175 models](/models?other=base_model:adapter:mistralai/Mistral-7B-v0.1)

Finetunes

[771 models](/models?other=base_model:finetune:mistralai/Mistral-7B-v0.1)

Merges

[115 models](/models?other=base_model:merge:mistralai/Mistral-7B-v0.1)

Quantizations

[165 models](/models?other=base_model:quantized:mistralai/Mistral-7B-v0.1)

## Spaces using  mistralai/Mistral-7B-v0.1100