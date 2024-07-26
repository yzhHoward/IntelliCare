import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.cluster import SpectralClustering
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import torch.distributed as dist

from pyhealth.datasets import SampleEHRDataset
from pyhealth.models import BaseModel
from pyhealth.models.utils import get_last_visit


class FinalAttentionQKV(nn.Module):
    def __init__(
        self,
        attention_input_dim: int,
        attention_hidden_dim: int,
        attention_type: str = "add",
        dropout: float = 0.5,
    ):
        super(FinalAttentionQKV, self).__init__()

        self.attention_type = attention_type
        self.attention_hidden_dim = attention_hidden_dim
        self.attention_input_dim = attention_input_dim

        self.W_q = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_k = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_v = nn.Linear(attention_input_dim, attention_hidden_dim)

        self.W_out = nn.Linear(attention_hidden_dim, 1)

        self.b_in = nn.Parameter(
            torch.zeros(
                1,
            )
        )
        self.b_out = nn.Parameter(
            torch.zeros(
                1,
            )
        )

        nn.init.kaiming_uniform_(self.W_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_v.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_out.weight, a=math.sqrt(5))

        self.Wh = nn.Parameter(
            torch.randn(2 * attention_input_dim, attention_hidden_dim)
        )
        self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
        self.ba = nn.Parameter(
            torch.zeros(
                1,
            )
        )

        nn.init.kaiming_uniform_(self.Wh, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))

        self.dropout = nn.Dropout(p=dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):

        (
            batch_size,
            time_step,
            input_dim,
        ) = input.size()  # batch_size * input_dim + 1 * hidden_dim(i)
        input_q = self.W_q(input[:, -1, :])  # b h
        input_k = self.W_k(input)  # b t h
        input_v = self.W_v(input)  # b t h

        if self.attention_type == "add":  # B*T*I  @ H*I

            q = torch.reshape(
                input_q, (batch_size, 1, self.attention_hidden_dim)
            )  # B*1*H
            h = q + input_k + self.b_in  # b t h
            h = self.tanh(h)  # B*T*H
            e = self.W_out(h)  # b t 1
            e = torch.reshape(e, (batch_size, time_step))  # b t

        elif self.attention_type == "mul":
            q = torch.reshape(
                input_q, (batch_size, self.attention_hidden_dim, 1)
            )  # B*h 1
            e = torch.matmul(input_k, q).squeeze()  # b t

        elif self.attention_type == "concat":
            q = input_q.unsqueeze(1).repeat(1, time_step, 1)  # b t h
            k = input_k
            c = torch.cat((q, k), dim=-1)  # B*T*2I
            h = torch.matmul(c, self.Wh)
            h = self.tanh(h)
            e = torch.matmul(h, self.Wa) + self.ba  # B*T*1
            e = torch.reshape(e, (batch_size, time_step))  # b t
        else:
            raise ValueError(
                "Unknown attention type: {}, please use add, mul, concat".format(
                    self.attention_type
                )
            )

        a = self.softmax(e)  # B*T
        if self.dropout is not None:
            a = self.dropout(a)
        v = torch.matmul(a.unsqueeze(1), input_v).squeeze()  # B*I

        return v, a


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-7):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SingleAttention(nn.Module):
    def __init__(
        self,
        attention_input_dim,
        attention_hidden_dim,
        attention_type="add",
        time_aware=False,
    ):
        super(SingleAttention, self).__init__()

        self.attention_type = attention_type
        self.attention_hidden_dim = attention_hidden_dim
        self.attention_input_dim = attention_input_dim
        self.time_aware = time_aware

        # batch_time = torch.arange(0, batch_mask.size()[1], dtype=torch.float32).reshape(1, batch_mask.size()[1], 1)
        # batch_time = batch_time.repeat(batch_mask.size()[0], 1, 1)

        if attention_type == "add":
            if self.time_aware:
                # self.Wx = nn.Parameter(torch.randn(attention_input_dim+1, attention_hidden_dim))
                self.Wx = nn.Parameter(
                    torch.randn(attention_input_dim, attention_hidden_dim)
                )
                self.Wtime_aware = nn.Parameter(torch.randn(1, attention_hidden_dim))
                nn.init.kaiming_uniform_(self.Wtime_aware, a=math.sqrt(5))
            else:
                self.Wx = nn.Parameter(
                    torch.randn(attention_input_dim, attention_hidden_dim)
                )
            self.Wt = nn.Parameter(
                torch.randn(attention_input_dim, attention_hidden_dim)
            )
            self.bh = nn.Parameter(
                torch.zeros(
                    attention_hidden_dim,
                )
            )
            self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
            self.ba = nn.Parameter(
                torch.zeros(
                    1,
                )
            )

            nn.init.kaiming_uniform_(self.Wd, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wx, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wt, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))
        elif attention_type == "mul":
            self.Wa = nn.Parameter(
                torch.randn(attention_input_dim, attention_input_dim)
            )
            self.ba = nn.Parameter(
                torch.zeros(
                    1,
                )
            )

            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))
        elif attention_type == "concat":
            if self.time_aware:
                self.Wh = nn.Parameter(
                    torch.randn(2 * attention_input_dim + 1, attention_hidden_dim)
                )
            else:
                self.Wh = nn.Parameter(
                    torch.randn(2 * attention_input_dim, attention_hidden_dim)
                )

            self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
            self.ba = nn.Parameter(
                torch.zeros(
                    1,
                )
            )

            nn.init.kaiming_uniform_(self.Wh, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))

        elif attention_type == "new":
            self.Wt = nn.Parameter(
                torch.randn(attention_input_dim, attention_hidden_dim)
            )
            self.Wx = nn.Parameter(
                torch.randn(attention_input_dim, attention_hidden_dim)
            )

            self.rate = nn.Parameter(torch.zeros(1) + 0.8)
            nn.init.kaiming_uniform_(self.Wx, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wt, a=math.sqrt(5))

        else:
            raise RuntimeError(
                "Wrong attention type. Please use 'add', 'mul', 'concat' or 'new'."
            )

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, input, mask, device):

        (
            batch_size,
            time_step,
            input_dim,
        ) = input.size()  # batch_size * time_step * hidden_dim(i)

        time_decays = (
            torch.tensor(range(time_step - 1, -1, -1), dtype=torch.float32)
            .unsqueeze(-1)
            .unsqueeze(0)
            .to(device=device)
        )  # 1*t*1
        b_time_decays = time_decays.repeat(batch_size, 1, 1) + 1  # b t 1

        if self.attention_type == "add":  # B*T*I  @ H*I
            last_visit = get_last_visit(input, mask)
            q = torch.matmul(last_visit, self.Wt)  # b h
            q = torch.reshape(q, (batch_size, 1, self.attention_hidden_dim))  # B*1*H
            if self.time_aware == True:
                k = torch.matmul(input, self.Wx)  # b t h
                time_hidden = torch.matmul(b_time_decays, self.Wtime_aware)  # b t h
            else:
                k = torch.matmul(input, self.Wx)  # b t h
            h = q + k + self.bh  # b t h
            if self.time_aware:
                h += time_hidden
            h = self.tanh(h)  # B*T*H
            e = torch.matmul(h, self.Wa) + self.ba  # B*T*1
            e = torch.reshape(e, (batch_size, time_step))  # b t
        elif self.attention_type == "mul":
            last_visit = get_last_visit(input, mask)
            e = torch.matmul(last_visit, self.Wa)  # b i
            e = (
                torch.matmul(e.unsqueeze(1), input.permute(0, 2, 1)).reshape(
                    batch_size, time_step
                )
                + self.ba
            )  # b t
        elif self.attention_type == "concat":
            last_visit = get_last_visit(input, mask)
            q = last_visit.unsqueeze(1).repeat(1, time_step, 1)  # b t i
            k = input
            c = torch.cat((q, k), dim=-1)  # B*T*2I
            if self.time_aware:
                c = torch.cat((c, b_time_decays), dim=-1)  # B*T*2I+1
            h = torch.matmul(c, self.Wh)
            h = self.tanh(h)
            e = torch.matmul(h, self.Wa) + self.ba  # B*T*1
            e = torch.reshape(e, (batch_size, time_step))  # b t

        elif self.attention_type == "new":
            last_visit = get_last_visit(input, mask)
            q = torch.matmul(last_visit, self.Wt)  # b h
            q = torch.reshape(q, (batch_size, 1, self.attention_hidden_dim))  # B*1*H
            k = torch.matmul(input, self.Wx)  # b t h
            dot_product = torch.matmul(q, k.transpose(1, 2)).reshape(
                batch_size, time_step
            )  # b t
            denominator = self.sigmoid(self.rate) * (
                torch.log(2.72 + (1 - self.sigmoid(dot_product)))
                * (b_time_decays.reshape(batch_size, time_step))
            )
            e = self.relu(self.sigmoid(dot_product) / (denominator))  # b * t
        else:
            raise ValueError(
                "Wrong attention type. Plase use 'add', 'mul', 'concat' or 'new'."
            )

        if mask is not None:
            e = e.masked_fill(mask == 0, -1e9)
        a = self.softmax(e)  # B*T
        v = torch.matmul(a.unsqueeze(1), input).reshape(batch_size, input_dim)  # B*I

        return v, a


class SAFARILayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        static_dim: int = 0,
        hidden_dim: int = 128,
        num_head: int = 4,
        pe_hidden: int = 64,
        dropout: int = 0.5,
        n_clu=5
    ):
        super().__init__()

        # hyperparameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  # d_model
        self.num_head = num_head
        # self.output_dim = output_dim
        self.dropout = dropout
        self.static_dim = static_dim

        self.GRUs = nn.ModuleList(
            [
                nn.GRU(1, self.hidden_dim, batch_first=True)
                for _ in range(self.input_dim)
            ]
        )
        self.feature_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.FinalAttentionQKV = FinalAttentionQKV(
            self.hidden_dim,
            self.hidden_dim,
            attention_type="mul",
            dropout=self.dropout,
        )

        if self.static_dim > 0:
            self.demo_proj_main = nn.Linear(self.static_dim, self.hidden_dim)

        self.dropout = nn.Dropout(p=self.dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
        self.GCN_W1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.GCN_W2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output0 = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        feat2clu = np.random.randint(0, n_clu, size=input_dim)
        clu2feat = [[] for i in range(n_clu)]
        for i in range(input_dim):
            clu2feat[feat2clu[i]].append(i)

        #Graph Init
        if static_dim > 0:
            adj_mat = torch.zeros(input_dim+1, input_dim+1, device='cuda')
        else:
            adj_mat = torch.zeros(input_dim, input_dim, device='cuda')
        for clu_id, cur_clu in enumerate(clu2feat):
            for i in cur_clu:
                for j in cur_clu:
                    if i != j:
                        adj_mat[i][j] = 1

        for i in range(len(adj_mat)):
            adj_mat[i][i] = 1

        if static_dim > 0:
            for i in range(input_dim):
                adj_mat[i][input_dim] = 1
                adj_mat[input_dim][i] = 1
        self.adj_mat = nn.Parameter(adj_mat, requires_grad=False)

    def graph_update(self, feature_emb, sim_metric='euclidean', n_clu=5, feat2clu=None):
        input_dim=self.input_dim
        if self.static_dim > 0:
            adj_mat = torch.zeros(input_dim+1, input_dim+1, device='cuda')
        else:
            adj_mat = torch.zeros(input_dim, input_dim, device='cuda')
        eps = 1e-7

        if sim_metric == 'euclidean':
            feature_mean_emb = [None for i in range(input_dim)]
            for i in range(input_dim):
                feature_mean_emb[i] = torch.mean(feature_emb[:,i,:].squeeze(), dim=0).cpu().numpy()
            feature_mean_emb = np.array(feature_mean_emb)
            #print(feature_mean_emb.shape)
            
            if feat2clu is None:
                kmeans = MiniBatchKMeans(n_clusters=n_clu, init='random', n_init=2).fit(feature_mean_emb)
                feat2clu = kmeans.labels_
            
            clu2feat = [[] for i in range(n_clu)]
            for i in range(input_dim):
                clu2feat[feat2clu[i]].append(i)

            for clu_id, cur_clu in enumerate(clu2feat):
                for i in cur_clu:
                    for j in cur_clu:
                        if i != j:
                            cos_sim = np.dot(feature_mean_emb[i], feature_mean_emb[j])
                            cos_sim = cos_sim / max(eps, float(np.linalg.norm(feature_mean_emb[i]) * np.linalg.norm(feature_mean_emb[j])))
                            adj_mat[i][j] = torch.tensor(cos_sim, device='cuda')


        elif 'kernel' in sim_metric:
            kernel_mat = torch.zeros((input_dim, input_dim), device='cuda')
            sigma = 0
            for i in range(input_dim):
                for j in range(input_dim):
                    if sim_metric == 'rbf_kernel':
                        sample_dist = F.pairwise_distance(feature_emb[:,i,:], feature_emb[:,j,:], p=2)
                    if sim_metric == 'laplacian_kernel':
                        sample_dist = F.pairwise_distance(feature_emb[:,i,:], feature_emb[:,j,:], p=1)
                    sigma += torch.mean(sample_dist)
            
            sigma = sigma / (input_dim * input_dim)
            #sigma = feature_emb.size(-1)
        
            for i in range(input_dim):
                for j in range(input_dim):
                    if sim_metric == 'rbf_kernel':
                        sample_dist = F.pairwise_distance(feature_emb[:,i,:], feature_emb[:,j,:], p=2)
                        kernel_mat[i, j] = torch.mean(torch.exp(-(sample_dist * sample_dist) / (2 * (sigma**2))))
                    elif sim_metric == 'laplacian_kernel':
                        sample_dist = F.pairwise_distance(feature_emb[:,i,:], feature_emb[:,j,:], p=1)
                        kernel_mat[i, j] = torch.mean(torch.exp(-sample_dist / sigma))
            #print(kernel_mat)
            aff_mat = np.array(kernel_mat.cpu().detach().numpy())
            #print(aff_mat)
            
            if feat2clu is None:
                kmeans = SpectralClustering(n_clusters=n_clu, affinity='precomputed', n_init=20).fit(aff_mat)
                feat2clu = kmeans.labels_
            
            clu2feat = [[] for i in range(n_clu)]
            for i in range(input_dim):
                clu2feat[feat2clu[i]].append(i)

            for clu_id, cur_clu in enumerate(clu2feat):
                for i in cur_clu:
                    for j in cur_clu:
                        if i != j:
                            adj_mat[i][j] = torch.tensor(aff_mat[i][j], device='cuda')

        for i in range(len(adj_mat)):
            adj_mat[i][i] = 1

        if self.static_dim > 0:
            for i in range(input_dim):
                adj_mat[i][input_dim] = 1
                adj_mat[input_dim][i] = 1

        self.adj_mat.data = adj_mat
        # return adj_mat, feat2clu, clu2feat

    def safari_encoder(self, input, static=None, mask=None):
        # input shape [batch_size, timestep, feature_dim]
        B, T, I = input.shape

        if self.static_dim > 0:
            demo_main = self.feature_proj(self.relu(self.demo_proj_main(static)).unsqueeze(1))

        hs = []
        for i in range(I):
            hs.append(get_last_visit(self.GRUs[i](input[:, :, i].unsqueeze(-1))[0], mask))
        hs = torch.stack(hs, dim=1)  # B, I, T, H
        # index = (lens.reshape(-1, 1, 1, 1) - 1)  # B, 1, 1, 1
        # ht = torch.gather(hs, -2, index.repeat(1, self.input_dim, 1, self.hidden_dim))
        ht = self.feature_proj(hs.reshape(B, I, -1))
        if self.static_dim > 0:
            ht = torch.cat((ht, demo_main), 1)
        gcn_hidden = self.dropout(ht)
        
        gcn_hidden = self.relu(self.GCN_W1(torch.matmul(self.adj_mat, gcn_hidden)))
        gcn_contexts = self.relu(self.GCN_W2(torch.matmul(self.adj_mat, gcn_hidden)))
        clu_context = gcn_contexts[:,:,:]
        weighted_contexts = self.FinalAttentionQKV(clu_context)[0]
        output = self.relu(self.output0(self.dropout(weighted_contexts)))
        output = self.dropout(output)
        return output, ht

    def forward(
        self,
        x: torch.tensor,
        static: Optional[torch.tensor] = None,
        mask: Optional[torch.tensor] = None,
    ) -> Tuple[torch.tensor]:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, input_dim].
            static: a tensor of shape [batch size, static_dim].
            mask: an optional tensor of shape [batch size, sequence len], where
                1 indicates valid and 0 indicates invalid.

        Returns:
            output: a tensor of shape [batch size, fusion_dim] representing the
                patient embedding.
            decov: the decov loss value
        """
        # rnn will only apply dropout between layers
        # lens = torch.max(torch.where(mask, torch.arange(mask.size(1), dtype=torch.int64), torch.tensor(-1, dtype=torch.int64)), dim=1)[0] + 1
        out, ht = self.safari_encoder(x, static, mask)
        # out = self.dropout(out)
        return out, ht


class SAFARI(BaseModel):
    def __init__(
        self,
        dataset: SampleEHRDataset,
        feature_keys: List[str],
        label_key: str,
        mode: str,
        use_embedding: List[bool],
        static_key: Optional[str] = None,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        self.embedding_dim = embedding_dim
        self.use_embedding = use_embedding
        self.hidden_dim = hidden_dim

        # validate kwargs for ConCare layer
        if "feature_size" in kwargs:
            raise ValueError("feature_size is determined by embedding_dim")

        # the key of self.feat_tokenizers only contains the code based inputs
        self.feat_tokenizers = {}
        self.static_key = static_key
        self.label_tokenizer = self.get_label_tokenizer()
        # the key of self.embeddings only contains the code based inputs
        self.embeddings = nn.ModuleDict()
        # the key of self.linear_layers only contains the float/int based inputs
        self.linear_layers = nn.ModuleDict()

        self.static_dim = 0
        if self.static_key is not None:
            self.static_dim = self.dataset.input_info[self.static_key]["len"]

        self.safari = nn.ModuleDict()
        # add feature ConCare layers
        for idx, feature_key in enumerate(self.feature_keys):
            input_info = self.dataset.input_info[feature_key]
            # sanity check
            if input_info["type"] not in [str, float, int]:
                raise ValueError(
                    "ConCare only supports str code, float and int as input types"
                )
            elif (input_info["type"] == str) and (input_info["dim"] not in [2, 3]):
                raise ValueError(
                    "ConCare only supports 2-dim or 3-dim str code as input types"
                )
            elif (input_info["type"] == str) and (use_embedding[idx] == False):
                raise ValueError(
                    "ConCare only supports embedding for str code as input types"
                )
            elif (input_info["type"] in [float, int]) and (
                input_info["dim"] not in [2, 3]
            ):
                raise ValueError(
                    "ConCare only supports 2-dim or 3-dim float and int as input types"
                )

            # for code based input, we need Type
            # for float/int based input, we need Type, input_dim
            if use_embedding[idx]:
                self.add_feature_transform_layer(feature_key, input_info)
                self.safari[feature_key] = SAFARILayer(
                    input_dim=embedding_dim,
                    static_dim=self.static_dim,
                    hidden_dim=self.hidden_dim,
                    **kwargs,
                )
            else:
                self.safari[feature_key] = SAFARILayer(
                    input_dim=input_info["len"],
                    static_dim=self.static_dim,
                    hidden_dim=self.hidden_dim,
                    **kwargs,
                )

        output_size = self.get_output_size(self.label_tokenizer)
        self.fc = nn.Linear(len(self.feature_keys) * self.hidden_dim, output_size)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        The label `kwargs[self.label_key]` is a list of labels for each patient.

        Args:
            **kwargs: keyword arguments for the model. The keys must contain
                all the feature keys and the label key.

        Returns:
            A dictionary with the following keys:
                loss: a scalar tensor representing the final loss.
                loss_task: a scalar tensor representing the task loss.
                loss_decov: a scalar tensor representing the decov loss.
                y_prob: a tensor representing the predicted probabilities.
                y_true: a tensor representing the true labels.
        """
        patient_emb = []
        for idx, feature_key in enumerate(self.feature_keys):
            input_info = self.dataset.input_info[feature_key]
            dim_, type_ = input_info["dim"], input_info["type"]

            # for case 1: [code1, code2, code3, ...]
            if (dim_ == 2) and (type_ == str):
                x = self.feat_tokenizers[feature_key].batch_encode_2d(
                    kwargs[feature_key]
                )
                # (patient, event)
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                # (patient, event, embedding_dim)
                x = self.embeddings[feature_key](x)
                # (patient, event)
                mask = torch.any(x !=0, dim=2)

            # for case 2: [[code1, code2], [code3, ...], ...]
            elif (dim_ == 3) and (type_ == str):
                x = self.feat_tokenizers[feature_key].batch_encode_3d(
                    kwargs[feature_key]
                )
                # (patient, visit, event)
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                # (patient, visit, event, embedding_dim)
                x = self.embeddings[feature_key](x)
                # (patient, visit, embedding_dim)
                x = torch.sum(x, dim=2)
                # (patient, visit)
                mask = torch.any(x !=0, dim=2)

            # for case 3: [[1.5, 2.0, 0.0], ...]
            elif (dim_ == 2) and (type_ in [float, int]):
                x, mask = self.padding2d(kwargs[feature_key])
                # (patient, event, values)
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                # (patient, event, embedding_dim)
                if self.use_embedding[idx]:
                    x = self.linear_layers[feature_key](x)
                # (patient, event)
                mask = mask.bool().to(self.device)

            # for case 4: [[[1.5, 2.0, 0.0], [1.8, 2.4, 6.0]], ...]
            elif (dim_ == 3) and (type_ in [float, int]):
                x, mask = self.padding3d(kwargs[feature_key])
                # (patient, visit, event, values)
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                # (patient, visit, embedding_dim)
                x = torch.sum(x, dim=2)

                if self.use_embedding[idx]:
                    x = self.linear_layers[feature_key](x)
                # (patient, event)
                mask = mask[:, :, 0]
                mask = mask.bool().to(self.device)
            else:
                raise NotImplementedError

            if self.static_dim > 0:
                static = torch.tensor(
                    kwargs[self.static_key], dtype=torch.float, device=self.device
                )
                x, _ = self.safari[feature_key](x, static=static, mask=mask)
            else:
                x, _ = self.safari[feature_key](x, mask=mask)
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        # (patient, label_size)
        logits = self.fc(patient_emb)
        # obtain y_true, loss, y_prob
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        loss_task = self.get_loss_function()(logits, y_true)
        loss = loss_task
        y_prob = self.prepare_y_prob(logits)
        results = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            'logit': logits,
        }
        if kwargs.get('embed', False):
            results['embed'] = patient_emb
        return results

    @torch.no_grad()
    def update(self, epoch, dataloader):
        if epoch > 5:
            return
        self.eval()
        patient_emb = [[] for _ in self.feature_keys]
        for kwargs in dataloader:
            for idx, feature_key in enumerate(self.feature_keys):
                input_info = self.dataset.input_info[feature_key]
                dim_, type_ = input_info["dim"], input_info["type"]

                # for case 1: [code1, code2, code3, ...]
                if (dim_ == 2) and (type_ == str):
                    x = self.feat_tokenizers[feature_key].batch_encode_2d(
                        kwargs[feature_key]
                    )
                    # (patient, event)
                    x = torch.tensor(x, dtype=torch.long, device=self.device)
                    # (patient, event, embedding_dim)
                    x = self.embeddings[feature_key](x)
                    # (patient, event)
                    mask = torch.any(x !=0, dim=2)

                # for case 2: [[code1, code2], [code3, ...], ...]
                elif (dim_ == 3) and (type_ == str):
                    x = self.feat_tokenizers[feature_key].batch_encode_3d(
                        kwargs[feature_key]
                    )
                    # (patient, visit, event)
                    x = torch.tensor(x, dtype=torch.long, device=self.device)
                    # (patient, visit, event, embedding_dim)
                    x = self.embeddings[feature_key](x)
                    # (patient, visit, embedding_dim)
                    x = torch.sum(x, dim=2)
                    # (patient, visit)
                    mask = torch.any(x !=0, dim=2)

                # for case 3: [[1.5, 2.0, 0.0], ...]
                elif (dim_ == 2) and (type_ in [float, int]):
                    x, mask = self.padding2d(kwargs[feature_key])
                    # (patient, event, values)
                    x = torch.tensor(x, dtype=torch.float, device=self.device)
                    # (patient, event, embedding_dim)
                    if self.use_embedding[idx]:
                        x = self.linear_layers[feature_key](x)
                    # (patient, event)
                    mask = mask.bool().to(self.device)

                # for case 4: [[[1.5, 2.0, 0.0], [1.8, 2.4, 6.0]], ...]
                elif (dim_ == 3) and (type_ in [float, int]):
                    x, mask = self.padding3d(kwargs[feature_key])
                    # (patient, visit, event, values)
                    x = torch.tensor(x, dtype=torch.float, device=self.device)
                    # (patient, visit, embedding_dim)
                    x = torch.sum(x, dim=2)

                    if self.use_embedding[idx]:
                        x = self.linear_layers[feature_key](x)
                    # (patient, event)
                    mask = mask[:, :, 0]
                    mask = mask.bool().to(self.device)
                else:
                    raise NotImplementedError

                if self.static_dim > 0:
                    static = torch.tensor(
                        kwargs[self.static_key], dtype=torch.float, device=self.device
                    )
                    _, ht = self.safari[feature_key](x, static=static, mask=mask)
                else:
                    _, ht = self.safari[feature_key](x, mask=mask)
                if dist.is_initialized():
                    if dist.get_rank() == 0:
                        gathered_features = [torch.zeros_like(ht) for _ in range(dist.get_world_size())]
                    else:
                        gathered_features = None
                    dist.gather(ht, gathered_features, dst=0)
                    if dist.get_rank() == 0:
                        for h in gathered_features:
                            patient_emb[idx].append(h.cpu())
                else:
                    patient_emb[idx].append(ht.cpu())
            
        for idx, feature_key in enumerate(self.feature_keys):
           
            if dist.is_initialized():
                # ht = dist.gather(ht, dst=0)
                if dist.get_rank() == 0:
                    ht = torch.cat(patient_emb[idx])
                    if self.safari[feature_key].static_dim > 0:
                        ht = ht[:, :-1]
                    self.safari[feature_key].graph_update(ht)
                dist.broadcast(self.safari[feature_key].adj_mat.data, src=0)
            else:
                ht = torch.cat(patient_emb[idx])
                if self.safari[feature_key].static_dim > 0:
                    ht = ht[:, :-1]
                self.safari[feature_key].graph_update(ht)
