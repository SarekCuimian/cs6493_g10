# cs_6493_g10
CityU CS6493 Project Topic1

This repository contains evaluation pipelines and results for large language models (Qwen, DeepSeek) on math-focused datasets（**GSM8K**, **MATH-500** and **AIME_1983_2024**)


## Datasets

- **GSM8K(test set)**: Grade school math word problems.
- **MATH-500**: Benchmark dataset for high school-level math reasoning.
- **AIME Dataset**: American Invitational Mathematics Examination problems (1983–2024).

## Evaluated Models

- `qwen2.5-math-1.5b-instruct`
- `deepseek-r1-distill-qwen-1.5b`

## Methods 
- COT (Wei J, Wang X et al.)
- Self-Refine (Madaan A, Tandon N, Gupta P, et al.)
- Self-Consistency (Wang X, Wei J, Schuurmans D, et al.)

## Results
results/{method_name}\_{dataset_name}\_{model_name}.csv
