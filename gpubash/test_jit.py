import torch
from torch import Tensor
from typing import Optional, Dict

class SafeModule(torch.nn.Module):
    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Optional[Tensor]]:
        y: Optional[Tensor] = None

        y = data["a"] 
        y = y + 1
        
        return {"out": y}

m_safe = torch.jit.script(SafeModule())
print(m_safe({"a": torch.tensor([1.0, 2.0])}))
print(m_safe({}))
