import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

# load tracer util function
import importlib.util
spec = importlib.util.spec_from_file_location("tracer", "tracer.py")
tracer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracer)

# stablize random elements
torch.set_deterministic(True)
torch.manual_seed(1234)

# load input tensors
attn_weights = torch.load("modeling_t5:T5ForConditionalGeneration:T5Stack:T5Block:T5LayerSelfAttention:T5Attention:attn_weights_before_dropout_0.pt")
attn_weights = attn_weights.half().cuda()

# call dropout
output = F.dropout(
    attn_weights, p=0.1, training=True
)

# save output
tracer.save_tensor(output, 'dropout_output')
