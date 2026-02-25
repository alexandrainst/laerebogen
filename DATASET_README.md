---
license: apache-2.0
dataset_tags:
- instruction-following
task_categories:
- text-generation
language:
- da
size_categories:
- 1M<n<10M
---

# Lærebogen

An instruction-following dataset for Danish.

This dataset features 5 million examples of multi-turn conversations in Danish, designed
to train instruction-following models, with a commercially usable license.

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
[here](https://github.com/alexandrainst/laerebogen). We used the
[Qwen/Qwen3-235B-A22B-Instruct-2507-FP8](https://hf.co/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8)
model for all steps (except the manual Step 1).

### Step 1: Seed Generation

We started by generating a set of 176 Danish seed prompts and answers manually, adapted
from the English [Self-Instruct seed
prompts](https://doi.org/10.18653/v1/2023.acl-long.754) as well as prompts crowdsourced
as part of the EU Horizon project TrustLLM (grant agreement number 101135671). These
seed prompts can be found
[here](https://github.com/alexandrainst/laerebogen/blob/main/data/seed_tasks.jsonl).

### Step 2: Base Dataset Generation

With the seed prompts in hand, we used an improved version of the [Alpaca
recipe](https://github.com/tatsu-lab/stanford_alpaca) to generate an initial instruction
dataset with 1 million examples. The main differences were that we used structured
generation to ensure the correct outputs, and that we used MinHash deduplication rather
than ROUGE computations, as this was many orders of magnitude faster. This used the seed
prompts from the previous step as few-shot examples, and were filtered using filters
that checked that the generated examples were not too short or too long, were not too
similar to existing instructions, and did not contain prompt words.

### Step 3: Grammar Correction

The generated dataset was then grammar-corrected, which includes translation to Danish
in case this was necessary. I.e., if the instruction was specifically about translation
to a non-Danish language, then we don't translate the output, but in some cases the
model ended up generating non-Danish instructions/outputs, so these were translated to
Danish here.

### Step 4: Quality Improvement

A number of the generated examples were non-sensical or generally of low quality, so we
run the generated instructions through the model again, this time asking it to rewrite
the instructions to improve their quality, in case they were of low quality.

### Step 5: Evolving the Dataset

We next used the [Evol-Instruct recipe](https://doi.org/10.48550/arXiv.2304.12244) to
evolve the dataset for 4 generations. This process both makes the examples more complex
and diverse. All the new evolved examples were added to the dataset and shuffled with
the previous examples.

### Step 6: Adding Follow-Up Questions

Finally, we added 3 follow-up queries and answers to each of the examples in the
dataset.

## License

This dataset is licensed under the [Apache 2.0
license](https://www.apache.org/licenses/LICENSE-2.0), allowing the dataset to be used
for any purpose, including commercial purposes. The model that we used was also released
with this license.

## Creators and Funders

This dataset was created by [Dan Saattrup Smart](https://huggingface.co/saattrupdan) and
[Sofie Helene Bruun](https://hf.co/sofiehb) from the [Alexandra
Institute](https://alexandra.dk) as part of the [Danish Foundation Models
project](https://www.foundationmodels.dk). The project is funded by the Danish Research
Reserve as part of the national budget of Denmark for 2025, and consists of the
following partners:

- [Alexandra Institute](https://alexandra.dk)
- [University of Copenhagen](https://www.ku.dk)
- [Aarhus University](https://www.au.dk)
- [University of Southern Denmark](https://www.sdu.dk)
