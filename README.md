# Lærebogen

An instruction-following dataset for Danish.

This dataset features XXX examples of multi-turn conversations in Danish, designed to
train instruction-following models, with a commercially usable license.


## Quick Start

To install the dependencies and set up the project, you can run the following command:

```bash
make install
```

To then generate the dataset, you can run:

```bash
make dataset
```


## Dataset Structure

All examples in the dataset are structured as follows:

```json
{
  "messages": [
	{
		"role": "user",
		"content": "(...)"
	},
	{
		"role": "assistant",
		"content": "(...)"
	},
	{
		"role": "user",
		"content": "(...)"
	},
	(...)
	{
		"role": "assistant",
		"content": "(...)"
	}
}
```


## Dataset Generation Process

The dataset was created using several steps, each of which is described in detail in the
subsections below. The code base used for the dataset generation can be found
[here](https://github.com/alexandrainst/laerebogen).

### Step 1: Seed Generation

We started by generating a set of 176 Danish seed prompts and answers manually, adapted
from the English [Self-Instruct seed
prompts](https://doi.org/10.18653/v1/2023.acl-long.754) as well as prompts crowdsourced
as part of the EU Horizon project TrustLLM (grant agreement number 101135671). These
seed prompts can be found
[here](https://github.com/alexandrainst/laerebogen/blob/main/data/seed_tasks.jsonl).

### Step 2: Base Dataset Generation

With the seed prompts in hand, we used the [Alpaca
recipe](https://github.com/tatsu-lab/stanford_alpaca) to generate an initial instruction
dataset with 52,000 examples, based on the
[Gemma-3-27b-pt](https://hf.co/google/gemma-3-27b-pt) base decoder model. This used the
seed prompts from the previous step as few-shot examples, and were filtered using the
same filters as in the Alpaca recipe, with the additional filter that the generated
examples had to be in Danish, which was checked using the [Lingua language detection
package](https://github.com/pemistahl/lingua-py).

### Step 3: Grammar Correction

The generated dataset was then grammar-corrected using the
[Gemma-3-27b-it](https://hf.co/google/gemma-3-27b-it) instruction-tuned model. We used
the base model in the previous step to not have any instruction bias in the generation,
but such bias is not applicable in this step.

### Step 4: Quality Improvement

A number of the generated examples were non-sensical or generally of low quality, so we
run the generated instructions through
[Gemma-3-27b-it](https://hf.co/google/gemma-3-27b-it) again, this time asking it to
rewrite the instructions to improve their quality, in case they were of low quality.

### Step 5: Evolving the Dataset

We next used the [Evol-Instruct recipe](https://doi.org/10.48550/arXiv.2304.12244) to
evolve the dataset for 4 generations, using the
[Gemma-3-27b-it](https://hf.co/google/gemma-3-27b-it) model. This process both makes the
examples more complex and diverse. All the new evolved examples were added to the
dataset and shuffled with the previous examples.

### Step 6: Adding Follow-Up Questions

Finally, we added 3 follow-up queries and answers to each of the examples in the
dataset, again using the [Gemma-3-27b-it](https://hf.co/google/gemma-3-27b-it) model.


## License

This dataset is licensed under the [Gemma Terms of
Use](https://ai.google.dev/gemma/terms), which allows use for both commercial and
non-commercial purposes, provided that the dataset is not used to [cause
harm](https://ai.google.dev/gemma/prohibited_use_policy). Any modifications of the
dataset as well as models trained on it must also be shared under the same license.


## Creators and Funders

This dataset was created by [Dan Saattrup Smart](https://huggingface.co/saattrupdan)
from the [Alexandra Institute](https://alexandra.dk) as part of the [Danish Foundation
Models project](https://www.foundationmodels.dk). The project is funded by the Danish
Research Reserve as part of the national budget of Denmark for 2025, and consists of the
following partners:

- [Alexandra Institute](https://alexandra.dk)
- [University of Copenhagen](https://www.ku.dk)
- [Aarhus University](https://www.au.dk)
- [University of Southern Denmark](https://www.sdu.dk)
