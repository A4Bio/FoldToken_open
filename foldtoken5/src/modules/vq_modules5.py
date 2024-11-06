import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max, scatter_sum
import numpy as np
from torch_geometric.nn.pool import knn_graph




def round_ste(z):
    """Round with straight through gradients."""
    zhat = torch.round(z)
    return z + (zhat - z).detach()




class FSQ(nn.Module):
    """Quantizer."""

    def __init__(self, levels: list[int], embedding_dim, vq_dim, eps: float = 1e-3):
        super(FSQ, self).__init__()
        self._levels = levels
        self._eps = eps
        self._levels_np = np.asarray(levels)
        self._basis = np.concatenate(([1], np.cumprod(self._levels_np[:-1]))).astype(np.int64)  # 修改此处为 np.int64
        self._implicit_codebook = self.indexes_to_codes(torch.arange(self.codebook_size, dtype=torch.int64))
        self.proj = nn.Linear(embedding_dim, vq_dim)
        self.proj_inv = nn.Linear(vq_dim, embedding_dim)

    @property
    def num_dimensions(self) -> int:
        """Number of dimensions expected from inputs."""
        return len(self._levels)

    @property
    def codebook_size(self) -> int:
        """Size of the codebook."""
        return np.prod(self._levels)

    @property
    def codebook(self):
        """Returns the implicit codebook. Shape (prod(levels), num_dimensions)."""
        return self._implicit_codebook

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels_np - 1) * (1 - self._eps) / 2
        offset = np.where(self._levels_np % 2 == 1, 0.0, 0.5)
        shift = np.tan(offset / half_l)
        return torch.tanh(z + torch.tensor(shift, dtype=z.dtype, device=z.device)) * torch.tensor(half_l, dtype=z.dtype, device=z.device) - torch.tensor(offset, dtype=z.dtype, device=z.device)

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))

        # Renormalize to [-1, 1].
        half_width = self._levels_np // 2
        return quantized / torch.tensor(half_width, dtype=z.dtype, device=z.device)

    def _scale_and_shift(self, zhat_normalized):
        # Scale and shift to range [0, ..., L-1]
        half_width = self._levels_np // 2
        return (zhat_normalized * torch.tensor(half_width, dtype=zhat_normalized.dtype, device=zhat_normalized.device)) + torch.tensor(half_width, dtype=zhat_normalized.dtype, device=zhat_normalized.device)

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels_np // 2
        return (zhat - torch.tensor(half_width, dtype=zhat.dtype, device=zhat.device)) / torch.tensor(half_width, dtype=zhat.dtype, device=zhat.device)

    def codes_to_indexes(self, zhat: torch.Tensor) -> torch.Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.num_dimensions
        zhat = self._scale_and_shift(zhat)
        return (zhat * torch.tensor(self._basis, dtype=zhat.dtype, device=zhat.device)).sum(dim=-1).type(torch.int64)  # 修改此处为 torch.int64

    def indexes_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Inverse of `codes_to_indexes`."""
        indices = indices.unsqueeze(-1)
        codes_non_centered = np.mod(
            np.floor_divide(indices.cpu().numpy(), self._basis), self._levels_np
        )
        return self._scale_and_shift_inverse(torch.tensor(codes_non_centered, dtype=indices.dtype, device=indices.device))
    
    def forward(self, h_in):
        h = self.proj(h_in)
        quantized = self.quantize(h)
        vq_code = self.codes_to_indexes(quantized)
        ''' looking for the k-nearest codebook entries, and select one based on the probability '''
        
        quantized = self.proj_inv(quantized)
        # count = torch.bincount(vq_code)
        # utization = 1-(count==0).sum()/count.shape[0]
        return quantized, vq_code



class FSQHier(nn.Module):
    """Quantizer."""
    def __init__(self, levels, embedding_dim, condition_layer, eps: float = 1e-3):
        super(FSQHier, self).__init__()
        self._eps = 1e-3
        level_map = {20: [4,5],
                     256: [4,4,4,4],
                     4096: [4,4,4,4,4,4]}

        self._levels = [level_map[one] for one in levels]
        self._levels_np = [np.asarray(one) for one in self._levels]
        self._basis = [ np.concatenate(([1], np.cumprod(one[:-1]))).astype(np.int64)  for one in self._levels_np ]
        
        self._implicit_codebook = []
        for i in range(len(self._levels_np)):
            codebook = self.indexes_to_codes(torch.arange(self.codebook_size[i], dtype=torch.int64), i)
            self._implicit_codebook.append(codebook)
        
        self.enc = nn.ModuleDict(
            {f"enc{level}": nn.Linear(embedding_dim, len(self._levels[i])) for i, level in enumerate(levels)}
        )
        
        self.condition = nn.ModuleDict(
            {f"condition{level}": self.build_condition_layer(len(self._levels[i]),embedding_dim,condition_layer,10) for i, level in enumerate(levels)}
        )
        
        self.dec = nn.Linear(10, embedding_dim)
    
        

    def build_condition_layer(self, input_dim, hidden_dim, condition_layer, vq_dim):
        layers = [nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()]
        for _ in range(condition_layer - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, vq_dim))
        return nn.Sequential(*layers)
    
    
    @property
    def num_dimensions(self) -> int:
        """Number of dimensions expected from inputs."""
        return [len(one) for one in self._levels]

    @property
    def codebook_size(self) -> int:
        """Size of the codebook."""
        return [np.prod(one) for one in self._levels]

    @property
    def codebook(self):
        """Returns the implicit codebook. Shape (prod(levels), num_dimensions)."""
        return self._implicit_codebook

    def bound(self, z: torch.Tensor, idx) -> torch.Tensor:
        half_l = (self._levels_np[idx] - 1) * (1 - self._eps) / 2
        offset = np.where(self._levels_np[idx] % 2 == 1, 0.0, 0.5)
        shift = np.tan(offset / half_l)
        
        half_l = torch.tensor(half_l, dtype=z.dtype,device=z.device)
        shift = torch.tensor(shift, dtype=z.dtype, device=z.device)
        offset = torch.tensor(offset, dtype=z.dtype, device=z.device)
        h = torch.tanh(z + shift) * half_l - offset
        return h

    def quantize(self, z: torch.Tensor, idx) -> torch.Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z, idx))

        # Renormalize to [-1, 1].
        half_width = self._levels_np[idx] // 2
        return quantized / torch.tensor(half_width, dtype=z.dtype, device=z.device)

    def _scale_and_shift(self, zhat_normalized, idx):
        # Scale and shift to range [0, ..., L-1]
        half_width = self._levels_np[idx] // 2
        return (zhat_normalized * torch.tensor(half_width, dtype=zhat_normalized.dtype, device=zhat_normalized.device)) + torch.tensor(half_width, dtype=zhat_normalized.dtype, device=zhat_normalized.device)

    def _scale_and_shift_inverse(self, zhat, idx):
        half_width = self._levels_np[idx] // 2
        return (zhat - torch.tensor(half_width, dtype=zhat.dtype, device=zhat.device)) / torch.tensor(half_width, dtype=zhat.dtype, device=zhat.device)

    def codes_to_indexes(self, zhat: torch.Tensor, idx) -> torch.Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.num_dimensions[idx]
        zhat = self._scale_and_shift(zhat, idx)
        return (zhat * torch.tensor(self._basis[idx], dtype=zhat.dtype, device=zhat.device)).sum(dim=-1).type(torch.int64)  # 修改此处为 torch.int64

    def indexes_to_codes(self, indices: torch.Tensor, idx) -> torch.Tensor:
        """Inverse of `codes_to_indexes`."""
        indices = indices.unsqueeze(-1)
        codes_non_centered = np.mod(
            np.floor_divide(indices.cpu().numpy(), self._basis[idx]), self._levels_np[idx]
        )
        return self._scale_and_shift_inverse(torch.tensor(codes_non_centered, dtype=indices.dtype, device=indices.device), idx)

    def normalize(self, x):
        return x/(torch.norm(x, dim=-1, keepdim=True)+1e-6)
    
    def forward(self, h_in, levels=[20,256,4096]):
        level_list = [20,256,4096]
        p = torch.rand(h_in.shape[0], device = h_in.device)
        vq_code_all, quantized_all = 0, 0
        for i, level in enumerate(levels):
            idx = level_list.index(level)
            proj = getattr(self.enc, f'enc{level}')
            h = proj(h_in)
            quantized = self.quantize(h, idx)
            vq_code = self.codes_to_indexes(quantized, idx)
            
            condition = getattr(self.condition, f'condition{level}')
            
            feat_share = self.normalize(condition(quantized))

            
            select = (i/len(levels)<p) & (p<=(i+1)/len(levels))
            vq_code_all = vq_code_all + vq_code*select 
            quantized_all = quantized_all + feat_share*select[:,None]
            
        quantized_all = self.dec(quantized_all)
            
        return quantized_all, vq_code_all