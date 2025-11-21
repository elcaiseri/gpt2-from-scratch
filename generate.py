# generate.py

import argparse

from gpt import GPT2LMHeadModel
from gpt_encoder import GPT2Tokenizer


def load_model(model_name: str, from_pretrained: str):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(from_pretrained)
    return tokenizer, model


def generate_text(
    prompt: str, model, tokenizer, max_tokens: int, temperature: float, do_sample: bool
):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(
        input_ids, max_length=max_tokens, temperature=temperature, do_sample=do_sample
    )
    return tokenizer.decode(output_ids[0].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, help="Text prompt to start generation.")
    parser.add_argument(
        "--model_name", type=str, default="gpt2", help="Tokenizer model name."
    )
    parser.add_argument(
        "--from_pretrained", type=str, default="gpt2", help="Pretrained model source."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature."
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Use sampling instead of greedy decoding.",
    )
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_name, args.from_pretrained)
    generated_text = generate_text(
        args.prompt, model, tokenizer, args.max_tokens, args.temperature, args.do_sample
    )

    print(generated_text)


if __name__ == "__main__":
    main()

# python generate.py "Hello, how are " --model_name gpt2 --from_pretrained gpt2 --max_tokens 100 --temperature 0.7 --do_sample
