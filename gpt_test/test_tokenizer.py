import pytest

from gpt_encoder import GPT2Tokenizer


@pytest.fixture
def tokenizer():
    """Initialize a fresh GPT2Tokenizer."""
    vocab_dir = "../gpt_encoder/vocab.json"
    merge_dir = "../gpt_encoder/merges.txt"
    return GPT2Tokenizer(vocab_dir, merge_dir)


def test_encode(tokenizer):
    """Test that encoding works."""
    text = "Hello, world!"
    tokens = tokenizer.encode(text)
    assert isinstance(tokens, list), "Encoded tokens should be a list"
    assert len(tokens) > 0, "Encoded tokens should not be empty"


def test_decode(tokenizer):
    """Test that decoding works."""
    text = "Hello, world!"
    tokens = tokenizer.encode(text)
    decoded_text = tokenizer.decode(tokens)
    assert decoded_text == text, (
        f"Decoded text '{decoded_text}' does not match original '{text}'"
    )
