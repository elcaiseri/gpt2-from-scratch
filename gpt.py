import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------ HuggingFace Imports for Test------
from transformers import AutoModelForCausalLM

from gpt_config import GPT2Config
from gpt_models import CausalLMOutput
from gpt_utils import Conv1D, NewGELUActivation


class GPT2LMHeadModel(nn.Module):
    def __init__(self, vocab_size, n_ctx, n_embd, n_layer, n_head):
        super().__init__()
        self.config = GPT2Config(vocab_size, n_ctx, n_embd, n_layer, n_head)
        self.transformer = GPT2Model(vocab_size, n_ctx, n_embd, n_layer, n_head)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # After initializing both modules:
        self.lm_head.weight = self.transformer.wte.weight

    def forward(self, input_ids, position_ids=None) -> tuple | CausalLMOutput:
        hidden_states = self.transformer(input_ids, position_ids)
        logits = self.lm_head(hidden_states)
        return CausalLMOutput(logits=logits)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        # Load the model from Hugging Face's pretrained models
        auto_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)

        config = auto_model.config

        model = cls(
            config.vocab_size,
            config.n_ctx,
            config.n_embd,
            config.n_layer,
            config.n_head,
        )

        model.load_state_dict(auto_model.state_dict(), strict=True)

        return model

    @torch.no_grad()
    def generate(self, input_ids, max_length=50, temperature=0.7, do_sample=False):
        assert temperature > 0, "Temperature must be greater than 0"
        assert input_ids.dim() == 2, "Input IDs must be a 2D tensor"
        _, input_len = input_ids.size()
        assert input_len < max_length, "Prompt length exceeds maximum length"

        for _ in range(max_length - input_len):
            logits = self(input_ids)
            next_token_logits = logits[:, -1, :] / temperature

            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            input_ids = torch.cat((input_ids, next_token), dim=1)

        return input_ids

    def train(self):
        # TODO: Implement training logic
        pass


class GPT2Model(nn.Module):
    def __init__(self, vocab_size, n_ctx, n_embd, n_layer, n_head):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head

        # Input embeddings
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(n_ctx, n_embd)
        self.drop = nn.Dropout(0.1)

        # Transformer blocks
        self.h = nn.ModuleList([GPT2Block(n_embd, n_head) for _ in range(n_layer)])

        # Layer norm
        self.ln_f = nn.LayerNorm(n_embd, eps=1e-5, elementwise_affine=True)

    def forward(self, input_ids, position_ids=None):
        if position_ids is None:
            position_ids = torch.arange(
                input_ids.size(1), dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Input embeddings
        input_embeds = self.wte(input_ids) + self.wpe(position_ids)
        x = self.drop(input_embeds)

        # Transformer blocks
        for block in self.h:
            x = block(x)

        # Layer norm
        x = self.ln_f(x)

        return x


class GPT2Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, eps=1e-5, elementwise_affine=True)

        # Self-attention
        self.attn = GPT2Attention(n_embd, n_head)

        self.ln_2 = nn.LayerNorm(n_embd, eps=1e-5, elementwise_affine=True)

        # Feed-forward
        self.mlp = GPT2MLP(n_embd)

    def forward(self, x):
        # Self-attention
        x = x + self.attn(self.ln_1(x))

        # Feed-forward
        x = x + self.mlp(self.ln_2(x))

        return x


class GPT2Attention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.split_size = n_embd
        self.head_dim = n_embd // n_head

        # Input projection
        self.c_attn = Conv1D(3 * n_embd, n_embd)

        # Output projection
        self.c_proj = Conv1D(n_embd, n_embd)

        # Dropout
        self.attn_dropout = nn.Dropout(0.1)

        # Residual dropout
        self.resid_dropout = nn.Dropout(0.1)

    def _split_heads(self, x):
        """
        Split the last dimension into (n_head, head_dim) and transpose the result
        to get the shape (batch_size, n_head, seq_len, head_dim).
        """
        b, t, c = x.size()
        x = x.view(b, t, self.n_head, self.head_dim)
        return x.transpose(1, 2).contiguous()

    def _merge_heads(self, x):
        """
        Merge the last two dimensions (n_head, head_dim) into one dimension (n_embd).
        """
        b, n_head, t, head_dim = x.size()
        x = x.transpose(1, 2).contiguous().view(b, t, n_head * head_dim)
        return x

    def _attn(self, q, k, v):
        """
        Compute the attention scores and apply softmax.
        """
        # (batch_size, n_head) -- seq_len, head_dim x head_dim, seq_len -> (batch_size, n_head, seq_len, seq_len)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Add attention
        mask = torch.tril(torch.ones((1, 1, q.size(2), k.size(2)), device=q.device))
        attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        # (batch_size, n_head) -- seq_len, seq_len x seq_len, head_dim -> (batch_size, n_head, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)
        return attn_output

    def forward(self, x):
        """
        Forward pass through the attention layer.
        """
        # Input projection
        x = self.c_attn(x)

        # Split into heads
        query, key, value = x.split(self.split_size, dim=-1)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        # Compute attention
        attn_output = self._attn(query, key, value)

        # Merge heads
        attn_output = self._merge_heads(attn_output)

        # Output projection
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output


class GPT2MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = Conv1D(4 * n_embd, n_embd)
        self.c_proj = Conv1D(n_embd, 4 * n_embd)
        self.act = NewGELUActivation()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.c_proj(x)
        return x


if __name__ == "__main__":
    # Example usage
    model = GPT2LMHeadModel(
        vocab_size=50257, n_ctx=1024, n_embd=768, n_layer=12, n_head=12
    )
    input_ids = torch.randint(
        0, 50257, (1, 1024)
    )  # Batch size of 1, sequence length of 1024
    logits = model(input_ids)
    print(logits.shape)  # Should be (1, 1024, 50257)
