import torch
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.models.gpt2 import GPT2Model
from torch.distributions import Categorical
import numpy as np
import pickle
from utils.argparser import ws
def LCSSDistance(a, b):
    lena = len(a)
    lenb = len(b)
    c = [[0 for i in range(lenb + 1)] for j in range(lena + 1)]
    for i in range(lena):
        for j in range(lenb):
            if a[i] == b[j]:
                c[i + 1][j + 1] = c[i][j] + 1
            elif c[i + 1][j] > c[i][j + 1]:
                c[i + 1][j + 1] = c[i + 1][j]
            else:
                c[i + 1][j + 1] = c[i][j + 1]
    return c[lena - 1][lenb - 1]


def DTWDistance(G, s1, s2):
    DTW = {}

    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            a_lng, a_lat = G.nodes[s1[i]]["lng"], G.nodes[s1[i]]["lat"]
            b_lng, b_lat = G.nodes[s2[j]]["lng"], G.nodes[s2[j]]["lat"]
            dist = (a_lng - b_lng) ** 2 + (a_lat - b_lat) ** 2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

    return DTW[len(s1) - 1, len(s2) - 1] ** 0.5

class Planner(nn.Module):

    def _calculate_unit_dir_vec(self, ya, xa, yb, xb):
        denom = ((yb - ya) ** 2 + (xb - xa) ** 2) ** 0.5
        if denom == 0.:
            return (0., 0.)
        return ((yb - ya) / denom, (xb - xa) / denom)

    def __init__(self, G: nx.Graph, A: torch.Tensor,  device: torch.device, args, pretrain_path):
        super().__init__()
        self.max_decode_step = args.max_decode_step
        self.device = device
        self.omega = args.omega
        self.beta = args.beta
        self.max_deg = A.long().sum(1).max()
        self.n_vertex = A.shape[0]
        self.mask = torch.zeros(self.n_vertex, self.max_deg + 1).long().to(self.device)
        self.mask[torch.arange(self.n_vertex), A.sum(1, keepdim=False).long()] = 1
        self.mask.cumsum_(dim=-1)
        self.mask = self.mask[:, :-1].bool()
        self.G = G
        self.rl_ratio = args.rl_ratio
        self.locations = torch.zeros([self.n_vertex, 2]).to(self.device)
        for k in range(self.n_vertex):
            self.locations[k, 0], self.locations[k, 1] = G.nodes[k]["lng"], G.nodes[k]["lat"]

        self.v_to_ord = dict()  # v : dict from v to ord
        self.ord_to_v = dict()  # v : [] list of vertices
        val, ind = A.long().topk(self.max_deg, dim=1)
        for i in range(self.n_vertex):
            valid_ind = ind[i][val[i] == 1].cpu().tolist()
            self.v_to_ord[i] = dict(zip(valid_ind, list(range(len(valid_ind)))))
            self.ord_to_v[i] = valid_ind

        self.tv_dir = torch.zeros([self.n_vertex, self.n_vertex, 2]).to(self.device)
        self.adj_dir = torch.zeros(self.n_vertex, self.max_deg, 2).to(self.device)
        for k in range(self.n_vertex):
            xb_m_xa = self.locations[:, 0] - self.locations[k, 0]
            yb_m_ya = self.locations[:, 1] - self.locations[k, 1]
            denom = (xb_m_xa.square() + yb_m_ya.square()).sqrt()
            self.tv_dir[k, denom > 0, 0], self.tv_dir[k, denom > 0, 1] = (xb_m_xa / denom)[denom > 0], \
            (yb_m_ya / denom)[denom > 0]

            self.adj_dir[k, torch.arange(len(self.ord_to_v[k])), :] = self.tv_dir[k, self.ord_to_v[k], :]

        config = transformers.GPT2Config(vocab_size=1, n_embd=args.x_emb_dim, n_head=4, n_layer=args.L_T)
        self.transformer = GPT2Model(config).to(device)

        distance_dim = 50
        direction_dim = 50
        traffic_state_dim = 50
        self.distance_mlp = nn.Linear(1, distance_dim).to(self.device)
        self.direction_mlp = nn.Linear(self.max_deg, direction_dim).to(self.device)
        self.traffic_state_mlp = nn.Linear(self.max_deg, traffic_state_dim).to(self.device)
        hidden_dim = 100
        self.out_mlp = nn.Sequential(
            nn.Linear(args.x_emb_dim + distance_dim + direction_dim + traffic_state_dim + 1 + args.x_emb_dim, hidden_dim).to(self.device),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(0.5 * hidden_dim)).to(self.device),
            nn.ReLU(),
            nn.Linear(int(0.5 * hidden_dim), self.max_deg).to(self.device)
        )
        self.traffic_states = pickle.load(open(ws + "/processed_data/CityA_nbr_travel_time_dict.pkl/nbr_time_dict.pkl", "rb"))

        if pretrain_path is not None:
            node2vec = pickle.load(open(pretrain_path, "rb"))
            assert self.n_vertex == len(node2vec)
            if args.x_emb_dim != node2vec[0].shape[0]:
                print("Use pretrained embed dims")
            x_emb_dim = node2vec[0].shape[0]
            nodeemb = torch.zeros(self.n_vertex + 2, x_emb_dim)
            for k in node2vec:
                nodeemb[k] = torch.from_numpy(node2vec[k])
            self.x_embedding = nn.Embedding.from_pretrained(nodeemb, freeze=False).to(device)
        else:
            self.x_embedding = nn.Embedding(self.n_vertex + 2, args.x_emb_dim, padding_idx=self.n_vertex, device=device).to( self.device)

    def get_traffic_state(self, xs, day, start_ts):
        traffic_state = np.zeros((len(xs), len(xs[0]), self.max_deg))
        for x_id, (x, d, ts) in enumerate(zip(xs, day, start_ts)):
            for n_id, n in enumerate(x):
                traffic_state[x_id, n_id] = self.traffic_states[d][ts][n]
        return torch.tensor(traffic_state, dtype=torch.float32).to(self.device)

    def get_traffic_state_(self, xs, day, start_ts):
        traffic_state = np.zeros((len(xs), self.max_deg))
        for x_id, (x, d, ts) in enumerate(zip(xs, day, start_ts)):
            traffic_state[x_id] = self.traffic_states[d][ts][x]
        return torch.tensor(traffic_state, dtype=torch.float32).to(self.device)

    def test_lcs(self, nodes, segment_num, od):
        with torch.autograd.no_grad():
            oris = od[:, 0].tolist()
            dests = od[:, 1].tolist()
            paths_planned = []
            paths_planned.extend(self.plan(oris, dests))
            mean_lcs = 0.
            max_lcs = 0.
            for k, (planned_path, ground, seg_num_path) in enumerate(zip(paths_planned, nodes, segment_num)):
                lcs = LCSSDistance(planned_path, ground.tolist()[:int(seg_num_path.item())])
                mean_lcs += lcs
                max_lcs = max(max_lcs, lcs)
            mean_lcs /= len(nodes)
        return mean_lcs, max_lcs, paths_planned

    def test_dtw(self, nodes, segment_num, od):
        with torch.autograd.no_grad():
            oris = od[:, 0].tolist()
            dests = od[:, 1].tolist()
            paths_planned = []
            paths_planned.extend(self.plan(oris, dests))
            mean_dtw = 0.
            max_dtw = 0.
            for k, (planned_path, ground, seg_num_path) in enumerate(zip(paths_planned, nodes, segment_num)):
                dtw = DTWDistance(self.G, planned_path[:int(seg_num_path.item())], ground.tolist())
                mean_dtw += dtw
                max_dtw = max(max_dtw, max_dtw)
            mean_dtw /= len(nodes)
        return mean_dtw, max_dtw


    def forward(self, xs, segment_num, od, day, start_ts):
        dests = od[:, 1].long().to(self.device)
        xs = xs.to(self.device)

        batch_size, horizon = xs.shape
        xs_actions = []

        # convert next node to index
        for i, x in enumerate(xs):
            length = int(segment_num[i].item())
            x_sliced = x[:length]
            action_list = []
            for a, b in zip(x_sliced, x_sliced[1:]):
                action = self.v_to_ord[a.item()][b.item()]
                action_list.append(action)
            action_tensor = torch.Tensor(action_list).long().to(self.device)

            xs_actions.append(action_tensor)

        xs_padded_emb = self.x_embedding(xs)
        dests_emb = self.x_embedding(dests)
        origis = od[:, 0]

        attention_mask = torch.ones_like(xs).long()
        for k in range(batch_size):
            attention_mask[k, int(segment_num[k].item()):] = 0


        multi_sample_log_probs = []
        multi_sample_xs = torch.zeros([batch_size, self.max_decode_step]).long().to(self.device)
        greedy_xs = torch.zeros([batch_size, self.max_decode_step]).long().to(self.device)
        """
        1. MLE loss
        """
        transformer_outputs = self.transformer(inputs_embeds=xs_padded_emb, attention_mask=attention_mask)
        hidden = transformer_outputs['last_hidden_state']  # b h c = embed

        distances = (self.locations[xs] - self.locations[dests].unsqueeze(1)).abs().sum(dim=-1, keepdim=True) * 100
        distances_feature = self.distance_mlp(distances)
        directions = (self.adj_dir[xs] * self.tv_dir[xs, dests.unsqueeze(1)].unsqueeze(2)).sum(dim=-1, keepdim=False)
        directions = torch.masked_fill(directions, self.mask[xs], -1)
        directions_feature = self.direction_mlp(directions)  # b h dim
        neighbor_traffic_state = self.get_traffic_state(xs, day, start_ts)
        neighbor_traffic_state = torch.masked_fill(neighbor_traffic_state, self.mask[xs], -1)
        neighbor_traffic_state_feature = self.traffic_state_mlp(neighbor_traffic_state)
        start_ts_mle = start_ts.repeat(1, xs.shape[1]).unsqueeze(2).to(self.device)
        feed = torch.concat( [hidden, distances_feature, directions_feature, dests_emb.unsqueeze(1).repeat(1, horizon, 1), neighbor_traffic_state_feature, start_ts_mle], dim=-1)
        out_logits = self.out_mlp(feed)
        mle_loss = sum([F.cross_entropy(out_logits[k][:int(segment_num[k].item()) - 1], xs_actions[k], reduction="mean") for k in range(batch_size)])
        """
        2. sample routes from multivariate distribution
        """
        multi_sample_xs[:, 0] = origis.clone()
        multi_sample_stop = torch.zeros([batch_size]).bool().to(self.device)
        multi_sample_length = torch.ones([batch_size]).long().to(self.device) * (self.max_decode_step)
        day_sample = day.unsqueeze(2).repeat(1, xs.shape[1], 1).to(self.device)
        start_ts_sample = start_ts.unsqueeze(2).repeat(1, xs.shape[1], 1).to(self.device)
        for i in range(1, self.max_decode_step):
            prefix = multi_sample_xs[:, :i].clone()
            prefix_emb = self.x_embedding(prefix)
            transformer_outputs = self.transformer(
                inputs_embeds=prefix_emb,
            )
            hidden = transformer_outputs['last_hidden_state']
            hidden = hidden[:, -1, :]
            distances = (self.locations[prefix[:, -1]] - self.locations[dests]).square().sum(dim=-1, keepdim=True).sqrt() * 100
            distances_feature = self.distance_mlp(distances)
            directions = (self.adj_dir[prefix[:, -1]] * self.tv_dir[prefix[:, -1], dests].unsqueeze(1)).sum(dim=-1,  keepdim=False)
            directions = torch.masked_fill(directions, self.mask[prefix[:, -1]], -1)
            directions_feature = self.direction_mlp(directions)

            day_prefix_sample = day_sample[:, :i]
            start_ts_prefix_sample = start_ts_sample[:, :i]

            neighbor_traffic_state = self.get_traffic_state_(prefix[:, -1], day_prefix_sample[:, -1], start_ts_prefix_sample[:, -1])
            neighbor_traffic_state = torch.masked_fill(neighbor_traffic_state, self.mask[prefix[:, -1]], -1)
            neighbor_traffic_state_feature = self.traffic_state_mlp(neighbor_traffic_state)

            ts = start_ts_prefix_sample[:, -1]

            feed = torch.concat(
                [hidden, distances_feature, directions_feature, dests_emb, neighbor_traffic_state_feature, ts],
                dim=-1)

            out_logits_gpt = self.out_mlp(feed)
            out_logits_gpt = torch.masked_fill(out_logits_gpt, self.mask[prefix[:, -1]], value=-1e20)
            multi_sample_log_p = torch.log_softmax(out_logits_gpt, dim=1)
            multi_sample_probs = multi_sample_log_p.exp()
            multi_sample_dist = Categorical(probs=multi_sample_probs)
            actions = multi_sample_dist.sample()
            multi_sample_next_log_prob = multi_sample_dist.log_prob(actions)
            multi_sample_log_probs.append(multi_sample_next_log_prob) # batch_size, append数量为max_decode_step - 1
            multi_sample_xs[:, i] = torch.Tensor([self.ord_to_v[prefix[k, -1].item()][actions[k]] for k in range(batch_size)]).long().to(self.device)
            multi_sample_length[multi_sample_xs[:, i] == dests] = i + 1
            multi_sample_stop = multi_sample_stop | (multi_sample_xs[:, i] == dests)
            if multi_sample_stop.all():
                break
        multi_sample_xs_list = [multi_sample_xs[k, :multi_sample_length[k]].cpu().tolist() for k in range(batch_size)]
        multi_sample_xs_list_refined = self.refine(multi_sample_xs_list, dests.cpu().tolist())

        """
        3. predict routes by test arg max
        """
        with torch.autograd.no_grad():
            greedy_xs[:, 0] = origis.clone()
            greedy_stop = torch.zeros([batch_size]).bool().to(self.device)
            greedy_length = torch.ones([batch_size]).long().to(self.device) * self.max_decode_step
            for i in range(1, self.max_decode_step):
                prefix = greedy_xs[:, :i].clone()
                prefix_emb = self.x_embedding(prefix)
                transformer_outputs = self.transformer(
                    inputs_embeds=prefix_emb,
                )
                hidden = transformer_outputs['last_hidden_state']
                hidden = hidden[:, -1, :]
                distances = (self.locations[prefix[:, -1]] - self.locations[dests]).square().sum(dim=-1, keepdim=True).sqrt() * 100
                distances_feature = self.distance_mlp(distances)
                directions = (self.adj_dir[prefix[:, -1]] * self.tv_dir[prefix[:, -1], dests].unsqueeze(1)).sum(dim=-1, keepdim=False)
                directions = torch.masked_fill(directions, self.mask[prefix[:, -1]], -1)
                directions_feature = self.direction_mlp(directions)

                day_prefix_greedy = day_sample[:, :i]
                start_ts_prefix_greedy = start_ts_sample[:, :i]

                neighbor_traffic_state = self.get_traffic_state_(prefix[:, -1], day_prefix_greedy[:, -1], start_ts_prefix_greedy[:, -1])
                neighbor_traffic_state = torch.masked_fill(neighbor_traffic_state, self.mask[prefix[:, -1]], -1)
                neighbor_traffic_state_feature = self.traffic_state_mlp(neighbor_traffic_state)

                ts = start_ts_prefix_greedy[:, -1]

                feed = torch.concat( [hidden, distances_feature, directions_feature, dests_emb, neighbor_traffic_state_feature, ts], dim=-1)

                out_logits_gpt = self.out_mlp(feed)
                out_logits_gpt = torch.masked_fill(out_logits_gpt, self.mask[prefix[:, -1]], value=-1e20)
                gpt_probs = torch.softmax(out_logits_gpt, dim=-1)
                gpt_probs = gpt_probs / gpt_probs.sum(dim=1, keepdim=True)
                actions = torch.argmax(gpt_probs, 1)
                greedy_xs[:, i] = torch.Tensor([self.ord_to_v[prefix[k, -1].item()][actions[k]] for k in range(batch_size)]).long().to(self.device)
                greedy_length[greedy_xs[:, i] == dests] = i + 1
                greedy_stop = greedy_stop | (greedy_xs[:, i] == dests)
                if greedy_stop.all():
                    break
            greedy_xs_list = [greedy_xs[k, :greedy_length[k]].cpu().tolist() for k in range(batch_size)]
            greedy_xs_list_refined = self.refine(greedy_xs_list, dests.cpu().tolist())

        """
        4. evaluation by test objectives
        """
        len_decode_steps = len(multi_sample_log_probs)
        multi_sample_reward_all = []
        greedy_lcs_reward_all = []
        for k, (multi_sample_path, greedy_path, ground, seg_num_path) in enumerate(zip(multi_sample_xs_list_refined, greedy_xs_list_refined, xs, segment_num)):
            ground = ground.detach().cpu().numpy().tolist()
            multi_sample_lcs = LCSSDistance(multi_sample_path, ground[:int(seg_num_path.item())])
            greedy_lcs = LCSSDistance(greedy_path, ground[:int(seg_num_path.item())])

            multi_sample_dtw = DTWDistance(self.G, multi_sample_path, ground[:int(seg_num_path.item())])
            greedy_dtw = DTWDistance(self.G, greedy_path, ground[:int(seg_num_path.item())])

            multi_sample_reward_all.append(multi_sample_lcs * self.omega - multi_sample_dtw * self.beta)
            greedy_lcs_reward_all.append(greedy_lcs * self.omega- greedy_dtw * self.beta)

        log_prob_mask = torch.zeros([batch_size, len_decode_steps]).to(self.device)
        refined_sample_length = torch.zeros([batch_size]).to(self.device)
        for l in range(batch_size):
            valid_len = len(multi_sample_xs_list_refined[l]) - 1
            if valid_len == 0:
                valid_len = 1
            log_prob_mask[l][:valid_len] = 1
            refined_sample_length[l] = valid_len

        multi_sample_log_probs = torch.stack(multi_sample_log_probs, dim = 1).to(self.device)
        log_probs = multi_sample_log_probs * log_prob_mask.clone()
        multi_sample_log_probs = torch.sum(log_probs, dim = 1) / refined_sample_length.clone()
        policy_loss = - torch.mean(torch.tensor(np.array(multi_sample_reward_all) - np.array(greedy_lcs_reward_all)).float().to(self.device) * multi_sample_log_probs)

        return mle_loss  + policy_loss * self.rl_ratio

    def inference(self, origs, dests, day, start_ts):
        with torch.no_grad():

            if type(origs) is list:
                origs = torch.Tensor(origs).long().to(self.device)
            if type(dests) is list:
                dests = torch.Tensor(dests).long().to(self.device)
            origs_emb = self.x_embedding(origs)  # b c
            dests_emb = self.x_embedding(dests)  # b c
            batch_size, x_emb_dim = origs_emb.shape
            xs = torch.zeros([batch_size, self.max_decode_step]).long().to(self.device)
            xs[:, 0] = origs
            stop = torch.zeros([batch_size]).bool().to(self.device)
            actual_length = torch.ones([batch_size]).long().to(self.device) * self.max_decode_step

            day = day.unsqueeze(2).repeat(1, self.max_decode_step, 1).to(self.device)
            start_ts = start_ts.unsqueeze(2).repeat(1, self.max_decode_step, 1).to(self.device)
            for i in range(1, self.max_decode_step):
                prefix = xs[:, :i]
                prefix_emb = self.x_embedding(prefix)
                # proposal from transformer
                transformer_outputs = self.transformer(
                    inputs_embeds=prefix_emb,
                )
                hidden = transformer_outputs['last_hidden_state']
                hidden = hidden[:, -1, :]

                distances = (self.locations[prefix[:, -1]] - self.locations[dests]).square().sum(dim=-1,  keepdim=True).sqrt() * 100
                distances_feature = self.distance_mlp(distances)

                directions = (self.adj_dir[prefix[:, -1]] * self.tv_dir[prefix[:, -1], dests].unsqueeze(1)).sum(dim=-1, keepdim=False)
                directions = torch.masked_fill(directions, self.mask[prefix[:, -1]], -1)
                directions_feature = self.direction_mlp(directions)  # b dim

                day_prefix = day[:, :i]
                start_ts_prefix = start_ts[:, :i]

                neighbor_traffic_state = self.get_traffic_state_(prefix[:, -1], day_prefix[:, -1], start_ts_prefix[:, -1])
                neighbor_traffic_state = torch.masked_fill(neighbor_traffic_state, self.mask[prefix[:, -1]], -1)
                neighbor_traffic_state_feature = self.traffic_state_mlp(neighbor_traffic_state)

                ts = start_ts_prefix[:, -1]

                feed = torch.concat([hidden, distances_feature, directions_feature, dests_emb, neighbor_traffic_state_feature, ts], dim=-1)

                out_logits_gpt = self.out_mlp(feed)
                out_logits_gpt = torch.masked_fill(out_logits_gpt, self.mask[prefix[:, -1]], value=-1e20)
                gpt_probs = torch.softmax(out_logits_gpt, dim=-1)

                syntheised_probs =  gpt_probs
                syntheised_probs = syntheised_probs / syntheised_probs.sum(1, keepdim=True)
                actions = torch.argmax(syntheised_probs, 1)

                xs[:, i] = torch.Tensor( [self.ord_to_v[prefix[k, -1].item()][actions[k]] for k in range(batch_size)]).long().to(self.device)
                actual_length[xs[:, i] == dests] = i + 1
                stop = stop | (xs[:, i] == dests)
                if stop.all():
                    break

                xs_list = [xs[k, :actual_length[k]].cpu().tolist() for k in range(batch_size)]
                xs_list_refined = self.refine(xs_list, dests.cpu().tolist())

            return xs_list_refined

    def refine(self, paths, dests):

        refined_paths = []
        for k, path in enumerate(paths):
            destination = dests[k]
            cutted_path = path
            origin = path[0]
            showup = set()
            points_filtered = []
            for _, v in enumerate(cutted_path):
                if v not in showup:
                    showup.add(v)
                    points_filtered.append(v)
                else:
                    while points_filtered[-1] != v:
                        showup.discard(points_filtered[-1])
                        points_filtered.pop()
            if len(points_filtered) <= 2:
                points_filtered = [origin, destination]
            refined_paths.append(points_filtered)
        return refined_paths
