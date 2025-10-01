import math
import torch
from torch import nn, Tensor
from torch.autograd import Variable
from torch.nn.init import xavier_normal_, constant_
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv, GCNConv
from torch.nn import Module
import torch.nn.functional as F
import numpy as np




class GNN_Encoder(Module):
    def __init__(self, hidden_size, sample_size, gnn_dropout_prob):
        super(GNN_Encoder, self).__init__()
        self.hidden_size = hidden_size
        in_channels = hidden_channels = self.hidden_size
        self.num_layers = len(sample_size)
        self.dropout = nn.Dropout(gnn_dropout_prob)
        self.gcn = GCNConv(self.hidden_size, self.hidden_size)
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, normalize=True))
        for i in range(self.num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, normalize=True))


    def forward(self, x, adjs, attr):
        xs = []
        x_all = x
        if self.num_layers > 1:
            for i, (edge_index, e_id, size) in enumerate(adjs):
                weight = attr[e_id].view(-1).type(torch.float)

                x = x_all
                if len(list(x.shape)) < 2:
                    x = x.unsqueeze(0)
                x = self.gcn(x, edge_index, weight)
                # sage
                x_target = x[:size[1]]  # Target nodes are always placed first.
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                    x = self.dropout(x)
        else:
            edge_index, e_id, size = adjs.edge_index, adjs.e_id, adjs.size
            x = x_all
            x = self.dropout(x)
            weight = attr[e_id].view(-1).type(torch.float)
            if len(list(x.shape)) < 2:
                x = x.unsqueeze(0)
            x = self.gcn(x, edge_index, weight)
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[-1]((x, x_target), edge_index)
        xs.append(x)
        return torch.cat(xs, 0)






class GCL4SR(nn.Module):
    def __init__(self, user_num, item_num, hidden_size, max_seq_length, num_attention_heads, global_graph, num_hidden_layers, lam1, lam2, sample_size):
        super(GCL4SR, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.hidden_size = hidden_size
        self.sample_size = sample_size
        self.max_seq_length = max_seq_length
        self.lam1 = lam1
        self.lam2 = lam2

        if torch.cuda.is_available():
            print("Using GPU")
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            print("Using Apple Silicon GPU")
            self.device = torch.device('mps')
        else:
            print("Using CPU")
            self.device = torch.device('cpu')
        self.global_graph = global_graph.to(self.device)
        self.global_gnn = GNN_Encoder(hidden_size, sample_size, gnn_dropout_prob=0.5)

        self.user_embeddings = nn.Embedding(user_num, hidden_size)
        self.item_embeddings = nn.Embedding(item_num, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_seq_length, hidden_size)

        # sequence encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size,
                                                        nhead=num_attention_heads,
                                                        dim_feedforward=4 * hidden_size,
                                                        dropout=0.5,
                                                        activation='gelu',
                                                        batch_first=True)
        
        self.item_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_hidden_layers)

        # AttNet
        self.w_1 = nn.Parameter(torch.Tensor(2*hidden_size, hidden_size))
        self.w_2 = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=False)

        self.w_g = nn.Linear(hidden_size, 1)
        self.w_e = nn.Linear(hidden_size, 1)

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.5)
        self.linear_transform = nn.Linear(3*hidden_size, hidden_size, bias=False)
        self.gnndrop = nn.Dropout(0.5)

        self.criterion = nn.CrossEntropyLoss()
        self.apply(self._init_weights)

        # user-specific gating
        self.gate_item = Variable(torch.zeros(hidden_size, 1).type
                                  (torch.FloatTensor), requires_grad=True).to(self.device)
        self.gate_user = Variable(torch.zeros(hidden_size, max_seq_length).type
                                  (torch.FloatTensor), requires_grad=True).to(self.device)
        self.gate_item = torch.nn.init.xavier_uniform_(self.gate_item)
        self.gate_user = torch.nn.init.xavier_uniform_(self.gate_user)


    def _init_weights(self, module):
        """ Initialize the weights """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)


    def gnn_encode(self, items):
        subgraph_loaders = NeighborSampler(self.global_graph.edge_index, node_idx=items, sizes=self.sample_size,
                                           shuffle=False,
                                           num_workers=0, batch_size=items.shape[0])
        g_adjs = []
        s_nodes = []
        for (b_size, node_idx, adjs) in subgraph_loaders:
            if type(adjs) == list:
                g_adjs = [adj.to(items.device) for adj in adjs]
            else:
                g_adjs = adjs.to(items.device)
            n_idxs = node_idx.to(items.device)
            s_nodes = self.item_embeddings(n_idxs).squeeze()
        attr = self.global_graph.edge_attr.to(items.device)
        g_hidden = self.global_gnn(s_nodes, g_adjs, attr)
        return g_hidden


    def final_att_net(self, seq_mask, hidden):
        batch_size = hidden.shape[0]
        lens = hidden.shape[1]
        pos_emb = self.position_embeddings.weight[:lens]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        seq_hidden = torch.sum(hidden * seq_mask, -2) / torch.sum(seq_mask, 1)
        seq_hidden = seq_hidden.unsqueeze(-2).repeat(1, lens, 1)
        item_hidden = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        item_hidden = torch.tanh(item_hidden)
        score = torch.sigmoid(self.linear_1(item_hidden) + self.linear_2(seq_hidden))
        att_score = torch.matmul(score, self.w_2)
        att_score_masked = att_score * seq_mask
        output = torch.sum(att_score_masked * hidden, 1)
        return output


    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)
        return mask


    def forward(self, data):
        user_ids = data[0]
        inputs = data[1]

        seq = inputs.flatten()
        seq_mask = (inputs == 0).float().unsqueeze(-1)
        seq_mask = 1.0 - seq_mask

        seq_hidden_global_a = self.gnn_encode(seq).view(-1, self.max_seq_length, self.hidden_size)
        seq_hidden_global_b = self.gnn_encode(seq).view(-1, self.max_seq_length, self.hidden_size)

        key_padding_mask = (inputs == 0)
        attn_mask = self.generate_square_subsequent_mask(self.max_seq_length).to(inputs.device)
        seq_hidden_local = self.item_embeddings(inputs)
        seq_hidden_local = self.LayerNorm(seq_hidden_local)
        seq_hidden_local = self.dropout(seq_hidden_local)

        seq_hidden_permute = seq_hidden_local
        encoded_layers = self.item_encoder(seq_hidden_permute,
                                           mask=attn_mask,
                                           src_key_padding_mask=key_padding_mask)
        sequence_output = encoded_layers

        user_emb = self.user_embeddings(user_ids).view(-1, self.hidden_size)

        gating_score_a = torch.sigmoid(torch.matmul(seq_hidden_global_a, self.gate_item.unsqueeze(0)).squeeze() +
                                       user_emb.mm(self.gate_user))
        user_seq_a = seq_hidden_global_a * gating_score_a.unsqueeze(2)
        gating_score_b = torch.sigmoid(torch.matmul(seq_hidden_global_b, self.gate_item.unsqueeze(0)).squeeze() +
                                       user_emb.mm(self.gate_user))
        user_seq_b = seq_hidden_global_b * gating_score_b.unsqueeze(2)

        user_seq_a = self.gnndrop(user_seq_a)
        user_seq_b = self.gnndrop(user_seq_b)

        hidden = torch.cat([sequence_output, user_seq_a, user_seq_b], -1)
        hidden = self.linear_transform(hidden)

        return sequence_output, hidden, user_seq_a, user_seq_b, (seq_hidden_global_a, seq_hidden_global_b), seq_mask


    def eval_stage(self, data):
        _, hidden, _, _, _, seq_mask = self.forward(data)
        hidden = self.final_att_net(seq_mask, hidden)
        return hidden


    def calculate_loss(self, data):
        targets = data[2]
        sequence_output, hidden, user_seq_a, user_seq_b, (seq_gnn_a, seq_gnn_b), seq_mask = self.forward(data)
        seq_out = self.final_att_net(seq_mask, hidden)
        seq_out = self.dropout(seq_out)
        test_item_emb = self.item_embeddings.weight[:self.item_num]
        logits = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        main_loss = self.criterion(logits, targets)

        sum_a = torch.sum(seq_gnn_a * seq_mask, 1) / torch.sum(seq_mask.float(), 1)
        sum_b = torch.sum(seq_gnn_b * seq_mask, 1) / torch.sum(seq_mask.float(), 1)

        info_hidden = torch.cat([sum_a, sum_b], 0)
        gcl_loss = self.GCL_loss(info_hidden, hidden_norm=True, temperature=0.5)

        seq_hidden_local = self.w_e(self.item_embeddings(data[1])).squeeze().unsqueeze(0)
        user_seq_a = self.w_g(user_seq_a).squeeze()
        user_seq_b = self.w_g(user_seq_b).squeeze()
        mmd_loss = self.MMD_loss(seq_hidden_local, user_seq_a) + self.MMD_loss(seq_hidden_local, user_seq_b)

        loss = main_loss + self.lam1 * gcl_loss + self.lam2 * mmd_loss
        return loss, main_loss, gcl_loss, mmd_loss
    

    def GCL_loss(self, hidden, hidden_norm=True, temperature=1.0):
        batch_size = hidden.shape[0] // 2
        LARGE_NUM = 1e9
        if hidden_norm:
            hidden = torch.nn.functional.normalize(hidden, p=2, dim=-1)
        hidden_list = torch.split(hidden, batch_size, dim=0)
        hidden1, hidden2 = hidden_list[0], hidden_list[1]

        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = torch.from_numpy(np.arange(batch_size)).to(hidden.device)
        masks = torch.nn.functional.one_hot(torch.from_numpy(np.arange(batch_size)).to(hidden.device), batch_size)

        logits_aa = torch.matmul(hidden1, hidden1_large.transpose(1, 0)) / temperature
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large.transpose(1, 0)) / temperature
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.transpose(1, 0)) / temperature
        logits_ba = torch.matmul(hidden2, hidden1_large.transpose(1, 0)) / temperature

        loss_a = torch.nn.functional.cross_entropy(torch.cat([logits_ab, logits_aa], 1), labels)
        loss_b = torch.nn.functional.cross_entropy(torch.cat([logits_ba, logits_bb], 1), labels)
        loss = (loss_a + loss_b)
        return loss
    

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)


    def MMD_loss(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        source = source.view(-1, self.max_seq_length)
        target = target.view(-1, self.max_seq_length)
        batch_size = int(source.size()[0])
        loss_all = []
        kernels = self.gaussian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        xx = kernels[:batch_size, :batch_size]
        yy = kernels[batch_size:, batch_size:]
        xy = kernels[:batch_size, batch_size:]
        yx = kernels[batch_size:, :batch_size]
        loss = torch.mean(xx + yy - xy - yx)
        loss_all.append(loss)
        return sum(loss_all) / len(loss_all)

