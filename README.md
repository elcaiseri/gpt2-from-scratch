# GPT-2 From Scratch with PyTorch

A clean implementation of GPT-2 from scratch using PyTorch, demonstrating core transformer architecture and text generation capabilities.

## Features

- Custom GPT-2 architecture implementation
- Text generation with configurable parameters
- Command-line interface
- Custom tokenizer and model configurations
- Comprehensive test suite

## Project Structure

```
gpt2-from-scratch/
├── generate.py          # Text generation CLI
├── gpt.py              # Core GPT-2 model
├── gpt_encoder/        # Tokenizer implementation
│   └── tokenizer.py    
├── gpt_config/         # Model configuration
│   └── config.py
└── gpt_test/          # Test suite
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers library (for tokenizer and weights loading)

## Installation

```bash
# Clone repository
git clone https://github.com/elcaiseri/gpt2-from-scratch.git
cd gpt2-from-scratch

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Text Generation

Generate text using the command-line interface:

#### Example

```bash
python generate.py "Hello, how are " \
    --model_name gpt2 \
    --from_pretrained gpt2 \
    --max_tokens 100 \
    --temperature 0.7 \
    --do_sample
```

### Command-line Options

| Option | Description | Default |
|--------|-------------|---------|
| `prompt` | Input text to start generation | Required |
| `--model_name` | Name of tokenizer model | "gpt2" |
| `--from_pretrained` | Pretrained model source | "gpt2" |
| `--max_tokens` | Maximum tokens to generate | 100 |
| `--temperature` | Sampling temperature (0.0-1.0) | 1.0 |
| `--do_sample` | Enable sampling mode | False |

### Running Tests

```bash
cd gpt_test
pytest
```

## Implementation Details

### Core Components

1. **GPT-2 Model** (`gpt.py`)
   - Multi-head self-attention
   - Position-wise feed-forward networks
   - Layer normalization
   - Residual connections

2. **Tokenizer** (`gpt_encoder/tokenizer.py`)
   - Custom implementation of GPT-2 tokenization
   - BPE (Byte Pair Encoding) support
   - Vocabulary management

3. **Configuration** (`gpt_config/config.py`)
   - Model hyperparameters
   - Architecture settings
   - Training configurations

## Development Status

- [x] Core model architecture
- [x] Text generation CLI
- [x] Basic tokenizer implementation
- [ ] Training pipeline
- [ ] Pre-trained weights support
- [ ] Model evaluation scripts

## References

- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [The Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/)
- [Transformers Library](https://github.com/huggingface/transformers)

## License

MIT License