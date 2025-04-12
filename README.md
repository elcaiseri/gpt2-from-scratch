# GPT-2 From Scratch with PyTorch

This project demonstrates building a GPT-2 model from scratch using PyTorch, inspired by the `transformers` library implementations.

## Features
- Custom GPT-2 architecture built with PyTorch.

#### TODO:
- Tokenization and training pipeline.
- Example usage for text generation.

## Requirements
- Python 3.8+
- PyTorch
- Transformers (optional for comparison)

## Usage
1. Clone the repository:
    ```bash
    git clone <repo-url>
    cd gpt2-from-scratch
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Test model
    ```bash
    cd gpt_test/
    pytest 
    ```
4. Model generation
    ```bash
    python generate.py "Hello, how are " --model_name gpt2 --from_pretrained gpt2 --max_tokens 100 --temperature 0.7 --do_sample
    ```



## References
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Transformers Library](https://github.com/huggingface/transformers)