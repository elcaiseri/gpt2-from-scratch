import pytest
import torch
import logging

from gpt import GPT2LMHeadModel

@pytest.fixture
def model():
    """Initialize a fresh GPT2LMHeadModel."""
    vocab_size = 50257
    n_ctx = 128  # Shorter context for faster testing
    n_embd = 768
    n_layer = 2  # fewer layers for quick testing
    n_head = 12
    return GPT2LMHeadModel(
        vocab_size=vocab_size,
        n_ctx=n_ctx,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head
    )


def test_forward_output_shape(model):
    """Test that model outputs correct shape."""
    vocab_size = 50257
    n_ctx = 128
    batch_size = 2
    input_ids = torch.randint(0, vocab_size, (batch_size, n_ctx))

    logits = model(input_ids)

    assert logits.shape == (batch_size, n_ctx, vocab_size), (
        f"Expected logits shape {(batch_size, n_ctx, vocab_size)}, got {logits.shape}"
    )


def test_from_pretrained_loads():
    """Test loading HuggingFace GPT2 weights into our model."""
    try:
        pretrained_model = GPT2LMHeadModel.from_pretrained("gpt2")
        assert pretrained_model is not None, "Pretrained model is None"
    except Exception as e:
        pytest.fail(f"from_pretrained failed: {e}")


def test_invalid_input(model):
    """Test that invalid input raises errors."""
    wrong_input = torch.randn(1, 128, 768)  # wrong shape: float instead of long ids

    with pytest.raises(RuntimeError):
        model(wrong_input)


def test_short_input(model):
    """Test model handles short input correctly."""
    vocab_size = 50257
    input_ids = torch.randint(0, vocab_size, (1, 10))  # very short input
    logits = model(input_ids)
    assert logits.shape == (1, 10, vocab_size), (
        f"Expected (1, 10, {vocab_size}), got {logits.shape}"
    )

def test_model_config(model):
    """Test that the model's config is set correctly."""
    config = model.config
    assert config.vocab_size == 50257, "Vocab size mismatch"
    assert config.n_ctx == 128, "Context size mismatch"
    assert config.n_embd == 768, "Embedding size mismatch"
    assert config.n_layer == 2, "Number of layers mismatch"
    assert config.n_head == 12, "Number of heads mismatch"