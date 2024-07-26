from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pyhealth.datasets import SampleEHRDataset
from pyhealth.models import BaseModel


class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MLPBlock(nn.Module):
    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            proj_drop=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm = norm_layer(dim) if norm_layer is not None else None
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(self, x):
        if self.norm is not None:
            x = x + self.mlp(self.norm(x))
        else:
            x = x + self.mlp(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        # Not a parameter
        self.register_buffer(
            "pos_table", self._get_sinusoid_encoding_table(n_position, d_hid)
        )

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, : x.size(-2)].clone().detach()


class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, proj_drop=0., qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BasicBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            proj_drop=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.position_enc = PositionalEncoding(dim, n_position=4)
        self.att_block = AttentionBlock(dim, num_heads, qkv_bias=qkv_bias, proj_drop=proj_drop)
        self.mlp = MLPBlock(dim=dim, mlp_ratio=mlp_ratio, proj_drop=proj_drop, 
                            act_layer=act_layer, norm_layer=norm_layer)

    def forward(self, x):
        x = self.position_enc(x)
        x = self.att_block(x)
        x = self.mlp(x)
        return x


class EncoderDrivenRectification(nn.Module):
    def __init__(
            self,
            query_dim,
            dim,
            num_heads=4,
            qkv_bias=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(query_dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, query, x):
        B, N, C = x.shape
        q = self.q(query).reshape(B, self.num_heads, 1, self.head_dim)
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)  # B, self.num_heads, N, self.head_dim
        x = torch.sigmoid((q @ k.transpose(-1, -2)) * self.scale) @ v
        x = x.transpose(1, 2).reshape(B, 1, -1)
        x = self.proj(x)
        return x


class PerplexityGuidedWeighting(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.x_proj = nn.Linear(dim, dim, bias=False)
        self.p_proj = nn.Linear(1, 1)
        
    def forward(self, x, perplexity):
        B, N, C = x.shape
        x = self.x_proj(x)
        perplexity = torch.tensor(perplexity).cuda()
        perplexity = 1 / torch.log(perplexity)
        perplexity = self.p_proj(perplexity.reshape(B, N, 1))
        x = (x * perplexity).mean(dim=1, keepdim=True)
        return x


class HybridAnalysisRefinement(nn.Module):
    def __init__(self, query_dim, hidden_dim):
        super().__init__()
        self.fusion = EncoderDrivenRectification(query_dim, hidden_dim)
        self.fusion1 = PerplexityGuidedWeighting(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.merge = nn.Linear(2 * hidden_dim, hidden_dim)
    
    def forward(self, query, text, ppl):
        query = query.reshape(query.shape[0], -1)
        text_features = self.fusion(query, text)
        text_features1 = self.fusion1(text, ppl)
        text_features = self.merge(torch.cat((self.norm(text_features), self.norm1(text_features1)), dim=-1))
        return text_features


class TextEmbed(nn.Module):
    def __init__(self, text_dim, hidden_dim=128, mlp_ratio=4):
        super().__init__()
        self.mlp = Mlp(
            in_features=text_dim,
            hidden_features=int(hidden_dim * mlp_ratio),
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=0.
        )
        
    def forward(self, text):
        text_features = self.mlp(torch.tensor(text).cuda())
        text_features = text_features.unsqueeze(1)
        return text_features


class IntelliCare(BaseModel):
    def __init__(self, backbone, dataset: SampleEHRDataset, label_key: str,
        mode: str, hidden_dim: int = 128, text_dim = 768, dropout=0.1):
        super().__init__(dataset=dataset, feature_keys=backbone.feature_keys,
            label_key=label_key, mode=mode)
        self.dropout = dropout
        self.backbone = backbone
        self.text_encoder = TextEmbed(text_dim, hidden_dim, mlp_ratio=4)
        self.hidden_dim = hidden_dim
        self.label_tokenizer = self.get_label_tokenizer()
        output_size = self.get_output_size(self.label_tokenizer)
        self.modal_fusion = BasicBlock(hidden_dim, 1, norm_layer=nn.LayerNorm, proj_drop=dropout)
        self.fusion = HybridAnalysisRefinement(len(self.feature_keys) * hidden_dim, hidden_dim)
        self.out = Mlp(len(self.feature_keys) * hidden_dim + hidden_dim, hidden_features=hidden_dim, out_features=output_size)

    def forward(self, embed=False, perplexity=None, **kwargs) -> Dict[str, torch.Tensor]:
        if 'data' in kwargs:  # for graphcare
            perplexity = kwargs['data']['perplexity']
            embedding = kwargs['data']['embedding']
        else:
            embedding = kwargs['embedding']
        kwargs["embed"] = True
        with torch.no_grad():
            results = self.backbone(**kwargs)
        patient_emb = results["embed"]
        y_true = results["y_true"]
        patient_emb = patient_emb.reshape(-1, len(self.feature_keys), self.hidden_dim)
        text_embedding = self.text_encoder(embedding)
        text_embedding = self.fusion(patient_emb, text_embedding, perplexity)
        patient_emb = torch.cat((patient_emb, text_embedding), dim=1)
        
        patient_emb = self.modal_fusion(patient_emb)
        logits = self.out(patient_emb.reshape(patient_emb.shape[0], -1))
        
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        results = {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
        if embed:
            results["embed"] = patient_emb
        return results
