import torch
import torch.utils.data
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F

class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout=0, output_layer=False):
        super(MultiLayerPerceptron, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            if dropout > 0:
                layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        return self.mlp(x)


class Regressor(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Regressor, self).__init__()
        self.linear_wide = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.linear_deep = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.linear_recurrent = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.out_layer = MultiLayerPerceptron(output_dim, (output_dim,), output_layer=True)

    def forward(self, deep, recurrent):
        fuse = self.linear_deep(deep) + self.linear_recurrent(recurrent)
        return self.out_layer(fuse)

class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear =nn.Linear(n_embed, num_experts)


    def forward(self, mh_output):
        logits = self.topkroute_linear(mh_output)

        noise_logits = self.noise_linear(mh_output)
        noise = torch.randn_like(logits)*F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices, F.softmax(logits, dim=-1)

class Expert(nn.Module):
    """ An MLP is a simple linear layer followed by a non-linearity i.e. each Expert """

    def __init__(self, n_embd):
        super().__init__()
        dropout = 0.1
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class SparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        gating_output, indices, softmax_gating_output = self.router(x)
        final_output = torch.zeros_like(x)

        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output, softmax_gating_output

class MoEUQ_network(torch.nn.Module):

    def __init__(self, args):
        super(MoEUQ_network, self).__init__()


        id_embed_dim = 20
        slice_dims = 145
        slice_embed_dim = 20
        mlp_out_dim = args.E_U
        lstm_hidden_size = args.E_U
        reg_input_dim = args.E_U
        reg_output_dim = args.E_U
        deep_mlp_dims = (args.E_U,)
        num_experts = args.C
        top_k = args.k
        n_embed = 128
        segment_dims =  12691 + 2
        node_dims = 4600 + 1
        self.distribution_embed = nn.Linear(args.m * 2 + 1 , n_embed)
        self.MoEUQ = SparseMoE(reg_input_dim, num_experts, top_k)

        self.segment_embedding = nn.Embedding(segment_dims, id_embed_dim)
        self.node_embedding = nn.Embedding(node_dims, id_embed_dim)

        self.slice_embedding = nn.Embedding(slice_dims, slice_embed_dim)
        self.all_mlp = nn.Sequential(
            nn.Linear(id_embed_dim + slice_embed_dim + n_embed , mlp_out_dim),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=mlp_out_dim, hidden_size=lstm_hidden_size, num_layers=2, batch_first=True)
        self.regressor = Regressor(reg_input_dim, reg_output_dim)
        self.regressor_lower = Regressor(reg_input_dim, reg_output_dim)
        self.regressor_upper = Regressor(reg_input_dim, reg_output_dim)
        self.deep_mlp = MultiLayerPerceptron(1 + id_embed_dim * 2, deep_mlp_dims)

    def forward(self, xs,  segment_travel_time, number_of_roadsegments, start_ts_10min, od, device):

        o_embed = self.node_embedding(od[:, 0].unsqueeze(1).to(device))
        d_embed = self.node_embedding(od[:, 1].unsqueeze(1).to(device))
        deep_output = self.deep_mlp(torch.cat([start_ts_10min.float().to(device), o_embed.squeeze(1).to(device), d_embed.squeeze(1).to(device)],  dim=-1))

        all_id_embedding = self.segment_embedding(xs.to(device))
        all_slice_embedding = self.slice_embedding(start_ts_10min.unsqueeze(1).expand(-1, xs.shape[1], -1).long().to(device).squeeze(-1))

        all_real = self.distribution_embed(segment_travel_time.to(device))  # [bs, seq, n_embd]
        all_input = torch.cat([all_id_embedding, all_slice_embedding, all_real], dim=2)

        recurrent_input = self.all_mlp(all_input)

        packed_all_input = pack_padded_sequence(recurrent_input, number_of_roadsegments.reshape(-1).cpu(), enforce_sorted=False, batch_first=True)
        seq_out, _ = self.lstm(packed_all_input)

        seq_out, _ = pad_packed_sequence(seq_out, batch_first=True)
        B, N_valid = seq_out.shape[0], seq_out.shape[1]
        seq_out, softmax_gating_output = self.MoEUQ(seq_out)

        mask_indices = torch.arange(N_valid).unsqueeze(0).expand(B, -1)
        mask = (mask_indices < number_of_roadsegments).unsqueeze(-1).float()
        seq_out = torch.sum(seq_out * mask.to(seq_out.device), dim=1)

        hat_y = self.regressor(deep_output, seq_out)
        bias_upper = self.regressor_upper(deep_output, seq_out)
        bias_lower = self.regressor_lower(deep_output, seq_out)

        valid_gating_output = softmax_gating_output * mask.to(seq_out.device)
        expert_load = torch.sum(valid_gating_output, dim=(0, 1))
        total_load = torch.sum(expert_load)
        normalized_load = expert_load / (total_load + 1e-9)
        num_experts = softmax_gating_output.shape[-1]
        ideal_load = 1.0 / num_experts
        load_balancing_loss = torch.sum(normalized_load * torch.log(normalized_load / ideal_load + 1e-9)) # normalization term for gating

        return hat_y, bias_lower, bias_upper, load_balancing_loss
