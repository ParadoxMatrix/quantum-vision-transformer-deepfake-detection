import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit.visualization import *
from qiskit_aer import AerSimulator

from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_aer.primitives import Sampler

from qiskit_algorithms.utils import algorithm_globals

import utils as U

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

import circuits as C

class quantumAttentionBlock(nn.Module):

    def __init__(self, vec_loader, matrix_mul, embed_dim, hidden_dim, num_heads, num_patches, dropout=0.0):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)

        self.embed_dim = embed_dim
        self.num_patches = num_patches + 1  # +1 for the class token vector

        aersim = AerSimulator(method='statevector', device='GPU')
        sampler = Sampler()
        sampler.set_options(backend=aersim)

        self.vx = C.Vx(embed_dim, vec_loader, matrix_mul)
        qc, num_weights = self.vx()
        self._vx = TorchConnector(SamplerQNN(circuit=qc, input_params=qc.parameters[-embed_dim+1:], weight_params=qc.parameters[:num_weights], input_gradients=True, sampler=sampler))

        qc, num_weights = C.xWx(embed_dim, vec_loader, matrix_mul)()
        self._xwx = TorchConnector(SamplerQNN(circuit=qc, input_params=qc.parameters[-embed_dim+1:]+qc.parameters[:embed_dim-1], weight_params=qc.parameters[embed_dim-1:-embed_dim+1], input_gradients=True, sampler=sampler))

        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        B, T, _ = inp_x.shape
        vx = torch.zeros(B, T, self.embed_dim, device=device)
        xwx = torch.zeros(B, T, T, device=device)

        for i in range(T):
            parameters = self.vx.get_RBS_parameters(inp_x[:, i, :])
            vx[:, i, :] = torch.sum(self._vx(parameters)[:, self.ei], dim=1)

        for i in range(T):
            for j in range(T):
                p = torch.cat((parameters[:, i, :], parameters[:, j, :]), dim=1)
                xwx[:, i, j] = torch.sum(self._xwx(p)[:, self.ei], dim=1)

        vx = torch.sqrt(vx + 1e-8)
        xwx = torch.sqrt(xwx + 1e-8)

        attn = F.softmax(xwx, dim=-1)
        t = torch.matmul(attn, vx.transpose(1, 2))
        x = x + t

        x = x + self.linear(self.layer_norm_2(x))
        return x

class QuantumVisionTransformer(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size, num_patches, dropout=0.0, vec_loader='diagonal', matrix_mul='pyramid'):
        super().__init__()

        self.patch_size = patch_size

        self.input_layer = nn.Linear(num_channels * (patch_size ** 2), embed_dim)

        self.transformer = nn.Sequential(*[quantumAttentionBlock(vec_loader, matrix_mul, embed_dim, hidden_dim, num_heads, num_patches, dropout=dropout) for _ in range(num_layers)])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))

    def forward(self, x):
        x = U.img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, :T+1]

        x = self.dropout(x)
        x = self.transformer(x)

        cls = x[:, 0, :]
        out = self.mlp_head(cls)
        return out
