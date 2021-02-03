import torch
from torch import nn
from torch.nn.parameter import Parameter


wo = nn.Linear(in_features=4096, out_features=1024, bias=False).half()

weight = Parameter(torch.load("Linear_wo_weight.pt").half())
tensor_name = "weight"
if not torch.isfinite(weight).all():
    print(tensor_name, "False")
    exit()
else:
    print(tensor_name, "True")


hidden_state = torch.load("hidden_states_before_Linear_wo.pt").half().cuda()
tensor_name = "hidden_state"
if not torch.isfinite(hidden_state).all():
    print(tensor_name, "False")
    exit()
else:
    print(tensor_name, "True")

wo.weight = weight
output = wo(hidden_state)
tensor_name = "output"
if not torch.isfinite(output).all():
    print(tensor_name, "False")
    exit()
else:
    print(tensor_name, "True")
