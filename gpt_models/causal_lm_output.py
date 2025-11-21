from typing import Optional

import torch
from pydantic import BaseModel, ConfigDict


class CausalLMOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor
