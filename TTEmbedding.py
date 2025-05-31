# @title TTLinear
# Tensor Train embedding https://arxiv.org/pdf/1901.10787
import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_einsum(num_tensors):
    a = 97
    R = chr(a+25) # 'z'
    lhs = [chr(a)+R]
    for i in range(1, num_tensors-1):lhs.append(R+chr(a+i)+R)
    lhs.append(R+chr(a+num_tensors-1))
    return ','.join(lhs) + '->' + ''.join([chr(a+i) for i in range(num_tensors)]) # az,zbz,zcz,zd->abcd

class TTLinear(nn.Module):
    def __init__(self, in_features=None, out_features=None, rank=256, std=1):
        super().__init__()
        self.lfeat = len(in_features)
        if self.lfeat==1: lst = in_features + out_features
        elif self.lfeat>=2: lst = [i*j for i, j in zip(in_features, out_features)]
        last = len(lst)
        var = last/rank**(1/(2*(std**.5)*last))
        c=1/last
        self.params = nn.ParameterList([nn.Parameter(torch.randn(lst[0], rank).clamp(-c,c)*var),
            *[nn.Parameter(torch.randn(rank, ij, rank).clamp(-c,c)*var) for ij in lst[1:-1]],
            nn.Parameter(torch.randn(rank, lst[-1]).clamp(-c,c)*var)])
        self.einsum_str = make_einsum(last)
        self.shape = [p for ij in zip(in_features, out_features) for p in ij]
        self.permute = list(range(0, 2*self.lfeat - 1, 2)) + list(range(1, 2*self.lfeat, 2))
    def weight(self): return torch.einsum(self.einsum_str, *self.params).reshape(self.shape).permute(self.permute).flatten(0,self.lfeat-1).flatten(1)

    def forward(self, x):
        weight = self.weight()
        return x.to(weight.dtype) @ weight


import math
class TTEmbedding(nn.Module):
    def __init__(self, in_dim, d_model, rank=256):
        super().__init__()
        self.ttlin = TTLinear(in_dim, d_model, rank, std=1) # https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        self.weight = self.ttlin.weight
        self.num_classes = math.prod(in_dim)

    def forward(self, x):
        return self.ttlin(F.one_hot(x, self.num_classes))
# self.out = lambda x: x @ self.tok_emb.weight().T  # weight tying

# in_features=(3,4,5,6); out_features=(2,3,4,5)
# in_features=[120]; out_features=[300]
# rank=16
# std=.5
# lin = TTLinear(in_features, out_features, rank, std).to(device)
# # x = torch.rand(4,math.prod((3,4,5,6)))
# x = torch.rand(4,7,math.prod(in_features), device=device)
# print(lin.params[0].device)
# out = lin(x)
# print(out.shape)
# print(lin.ttlin.params[0].device)

# emb = TTEmbedding(in_features, out_features, rank).to(device)
# x = torch.randint(0, math.prod(in_features), (2, 5), device=device)
# out = emb(x)
# print(out.shape)

# o=lin.weight
# print(o.mean().item(), o.std().item(), o.min().item(), o.max().item())

# import matplotlib.pyplot as plt
# plt.rcParams["figure.figsize"] = (4,4)
# # plt.hist(o.flatten().tolist(), bins=20, alpha=.5, label='context mask')
# # plt.hist(o[:100,:100].flatten().tolist(), bins=20, alpha=.5, label='context mask')
# x = torch.randn(100,100)*std
# # plt.hist(x.flatten().tolist(), bins=20, alpha=.5, label='context mask')
# plt.hist([o[:100,:100].flatten().tolist(), x.flatten().tolist()], bins=20, alpha=.5, label='context mask')
# plt.show()
