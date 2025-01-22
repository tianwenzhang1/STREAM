#!/usr/bin/python3
# coding: utf-8
# @Time    : 2020/11/5 10:27

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.dgl_gnn import UnsupervisedGAT, UnsupervisedGIN
from module.gps_transformer_layer import Encoder as Transformer
import dgl


def get_dict_info_batch(input_id, features_dict):
    """
    batched dict info
    """
    # input_id = [1, batch size]
    input_id = input_id.reshape(-1)
    features = torch.index_select(features_dict, dim=0, index=input_id)
    return features


def mask_log_softmax(x, mask, log_flag=True):
    maxes = torch.max(x, 1, keepdim=True)[0]
    x_exp = torch.exp(x - maxes) * mask
    x_exp_sum = torch.sum(x_exp, 1, keepdim=True)
    if log_flag:
        pred = x_exp / (x_exp_sum + 1e-6)
        pred = torch.clip(pred, 1e-6, 1)
        output_custom = torch.log(pred)
    else:
        output_custom = x_exp / (x_exp_sum + 1e-6)
    return output_custom


def mask_graph_log_softmax(g, log_flag=True):
    lg = g.ndata['lg']
    w = g.ndata['w']

    maxes = dgl.max_nodes(g, 'lg')
    maxes = dgl.broadcast_nodes(g, maxes)

    x_exp = torch.exp(lg - maxes) * w
    g.ndata['lg'] = x_exp
    x_exp_sum = dgl.sum_nodes(g, 'lg')
    x_exp_sum = dgl.broadcast_nodes(g, x_exp_sum)

    if log_flag:
        pred = x_exp / (x_exp_sum + 1e-6)
        pred = torch.clip(pred, 1e-6, 1)
        output_custom = torch.log(pred)
    else:
        output_custom = x_exp / (x_exp_sum + 1e-6)
    return output_custom


# add
class MLP2(nn.Module):
    """
    MLP with two output layers
    """
    def __init__(self, input_size, hidden_size, output_size,
                 dropout, use_selu=False):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, output_size)
        self.fc22 = nn.Linear(hidden_size, output_size)
        self.nonlinear_f = F.selu if use_selu else F.relu
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        h1 = self.dropout(self.nonlinear_f(self.fc1(x)))
        return self.fc21(h1), self.fc22(h1)

def logwsumexp(x, w):
    """
    log weighted sum (along dim 1) exp, i.e., log(sum(w * exp(x), 1)).

    Input:
      x (n, m): exponents
      w (n, m): weights
    Output:
      y (n,)
    """
    maxv, _ = torch.max(x, dim=1, keepdim=True)
    y = torch.log(torch.sum(torch.exp(x - maxv) * w, dim=1, keepdim=True)) + maxv
    return y.squeeze(1)

class ProbRho(nn.Module):
    """
    s1 (14): road type
    s2 (7): number of lanes
    s3 (2): one way or not
    """
    def __init__(self, num_u, dim_u, dict_u, lengths,
                       num_s1, dim_s1, dict_s1,
                       # num_s2, dim_s2, dict_s2,
                       # num_s3, dim_s3, dict_s3,
                       hidden_size, dim_rho,
                       dropout, use_selu, device):
        super(ProbRho, self).__init__()
        self.lengths = torch.tensor(lengths, dtype=torch.float32, device=device)
        self.dict_u = dict_u
        self.dict_s1 = dict_s1
        # self.dict_s2 = dict_s2
        # self.dict_s3 = dict_s3
        self.embedding_u = nn.Embedding(num_u, dim_u)
        self.embedding_s1 = nn.Embedding(num_s1, dim_s1)
        # self.embedding_s2 = nn.Embedding(num_s2, dim_s2)
        # self.embedding_s3 = nn.Embedding(num_s3, dim_s3)
        self.device = device
        # +dim_s2+dim_s3
        self.f = MLP2(dim_u+dim_s1, hidden_size, dim_rho, dropout, use_selu)

    def roads2u(self, roads):  # 将道路id映射到u的特征值
        """
        road id to word id (u)
        """
        return self.roads_s_i(roads, self.dict_u)

    def roads_s_i(self, roads, dict_s):
        """
        road id to feature id

        This function should be called in cpu
        ---
        Input:
        roads (batch_size * seq_len): road ids
        dict_s (dict): the mapping from road id to feature id
        Output:
        A tensor like roads
        """
        return roads.clone().apply_(lambda k: dict_s[k])

    def roads_length(self, roads, ratios=None):
        """
        roads (batch_size, seq_len): road id to road length
        ratios (batch_size, seq_len): The ratio of each road segment
        """
        if ratios is not None:
            return self.lengths[roads] * ratios.to(self.device)
        else:
            return self.lengths[roads]

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add(mu)
        else:
            return mu

    def forward(self, roads):
        """
        roads (batch_size * seq_len)
        """
        u  = self.embedding_u(self.roads2u(roads).to(self.device))
        s1 = self.embedding_s1(self.roads_s_i(roads, self.dict_s1).to(self.device))
        # s2 = self.embedding_s2(self.roads_s_i(roads, self.dict_s2).to(self.device))
        # s3 = self.embedding_s3(self.roads_s_i(roads, self.dict_s3).to(self.device))
        x = torch.cat([u, s1], dim=3)  #, s2, s3
        x = torch.squeeze(x, dim=2)
        mu, logvar = self.f(x)
        return self.reparameterize(mu, logvar)



class ProbTraffic(nn.Module):
    """
    Modelling the probability of the traffic state `c`
    """

    # def __init__(self, in_channels, out_channels, kernel_size, stride=1,
    #              padding=0, dilation=1, groups=1, bias=True)
    def __init__(self, n_in, hidden_size, dim_c, dropout, use_selu):
        super(ProbTraffic, self).__init__()
        conv_layers = [
            nn.Conv2d(n_in, 32, (5, 5), stride=2, padding=1),   # 二维卷积层
            nn.BatchNorm2d(32),    # 二维批归一化层，规范化卷积层的输出
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, (4, 4), stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, (4, 4), stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AvgPool2d(7)   # 二维平均池化层
        ]
        self.f1 = nn.Sequential(*conv_layers)
        self.f2 = MLP2(128*2*2, hidden_size, dim_c, dropout, use_selu)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add(mu)
        else:
            return mu

    def forward(self, T):
        """
        Input:
          T (batch_size, nchannel, height, width)
        Output:
          c, mu, logvar (batch_size, dim_c)
        """
        x = self.f1(T)
        mu, logvar = self.f2(x.view(x.size(0), -1))
        return self.reparameterize(mu, logvar), mu, logvar

class ProbTravelTime(nn.Module):
    def __init__(self, dim_rho, dim_c,
                       hidden_size, dropout, use_selu):
        super(ProbTravelTime, self).__init__()
        self.f = MLP2(dim_rho+dim_c, hidden_size, 128, dropout, use_selu)

    def forward(self, rho, c):
        """
        rho (batch_size, seq_len, dim_rho)
        c (1, dim_c): the traffic state vector sampling from ProbTraffic
        w (batch_size, seq_len): the normalized road lengths
        l (batch_size, ): route or path lengths
        """
        ## (batch_size, seq_len, dim_rho+dim_c)
        x = torch.cat([rho, c.expand(*rho.shape[:-1], -1)], 2)
        ## (batch_size, seq_len, 1)
        logm, logv = self.f(x)
        ## (batch_size, seq_len)
        return logm, logv


class RoadGNN(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.gnn_type = parameters.gnn_type
        self.node_input_dim = parameters.id_emb_dim
        self.node_hidden_dim = parameters.hid_dim
        self.num_layers = parameters.num_layers
        if self.gnn_type == 'gat':
            self.gnn = UnsupervisedGAT(self.node_input_dim, self.node_hidden_dim, edge_input_dim=0,
                                       num_layers=self.num_layers)
        else:
            self.gnn = UnsupervisedGIN(self.node_input_dim, self.node_hidden_dim, edge_input_dim=0,
                                       num_layers=self.num_layers)
        self.dropout = nn.Dropout(parameters.dropout)

    def forward(self, g, x, readout=True):
        '''
        :param x: road emb id with size [node size, id dim]
        :return: road hidden emb with size [graph size, hidden dim] if readout
                 else [node size, hidden dim]
        '''
        x = self.dropout(self.gnn(g, x))
        if not readout:
            return x

        g.ndata['x'] = x
        if 'w' in g.ndata:
            return dgl.mean_nodes(g, 'x', weight='w'), g
        else:
            return dgl.mean_nodes(g, 'x'), g


class Extra_MLP(nn.Module):
    """
        MLP with tanh activation function.
    """

    def __init__(self, parameters):
        super().__init__()
        self.pro_input_dim = parameters.pro_input_dim
        self.pro_output_dim = parameters.pro_output_dim
        self.fc_out = nn.Linear(self.pro_input_dim, self.pro_output_dim)

    def forward(self, x):
        out = torch.tanh(self.fc_out(x))
        return out


class Encoder(nn.Module):
    """
        Trajectory Encoder.
        Set online_feature_flag=False.
        Keep pro_features_flag (hours and holiday information).
        Encoder: RNN + MLP
    """

    def __init__(self, parameters):
        super().__init__()
        self.hid_dim = parameters.hid_dim
        self.pro_output_dim = parameters.pro_output_dim
        self.online_features_flag = parameters.online_features_flag
        self.dis_prob_mask_flag = parameters.dis_prob_mask_flag
        self.pro_features_flag = parameters.pro_features_flag
        self.device = parameters.device
        self.grid_flag = parameters.grid_flag
        self.transformer_layers = parameters.transformer_layers
        self.gnn_type = parameters.gnn_type
        self.traffic_features = parameters.traffic_features
        self.speed_flag = parameters.speed_features
        self.speed_dim = parameters.speed_dim

        input_dim = 3
        if self.traffic_features:
            input_dim += parameters.traffic_dim
        if self.online_features_flag:  # 实时特征
            input_dim += parameters.online_dim
        if self.dis_prob_mask_flag:
            input_dim += parameters.hid_dim
        if self.grid_flag:
            input_dim += parameters.id_emb_dim // 2

        self.fc_in = nn.Linear(input_dim, self.hid_dim)
        self.pred_out = nn.Linear(self.hid_dim, 1)

        self.transformer = Transformer(self.gnn_type, self.hid_dim, self.transformer_layers, self.speed_flag,
                                       self.device)
        # self.final_encoder = EncoderLayer(self.hid_dim)

        if self.pro_features_flag:
            self.extra = Extra_MLP(parameters)
            self.fc_hid = nn.Linear(self.hid_dim + self.pro_output_dim, self.hid_dim)

        if self.traffic_features:
            self.traffic_dim = parameters.traffic_dim
            self.rho = ProbRho(
                num_u=parameters.num_u, dim_u=parameters.dim_u, dict_u=parameters.dict_u, lengths=parameters.lengths,
                num_s1=parameters.num_s1, dim_s1=parameters.dim_s1, dict_s1=parameters.dict_s1,
                hidden_size=parameters.hidden_size1, dim_rho=parameters.dim_rho,
                dropout=parameters.traffic_dropout, use_selu=parameters.use_selu, device=self.device
            )
            self.c = ProbTraffic(
                n_in=parameters.n_in, hidden_size=parameters.hidden_size2, dim_c=parameters.dim_c,
                dropout=parameters.traffic_dropout, use_selu=parameters.use_selu
            )
            self.traffic = ProbTravelTime(
                dim_rho=parameters.dim_rho, dim_c=parameters.dim_c,
                hidden_size=parameters.hidden_size3, dropout=parameters.traffic_dropout, use_selu=parameters.use_selu
            )
    def forward(self, src, speed_cons, src_len,src_rids, data_S, g, pro_features):
        # src = [src len, batch size, 3]
        # if only input trajectory, input dim = 2; elif input trajectory + behavior feature, input dim = 2 + n
        # src_len = [batch size]
        max_src_len = src.size(0)
        bs = src.size(1)
        mask3d = torch.zeros(bs, max_src_len, max_src_len).to(self.device)
        mask2d = torch.zeros(bs, max_src_len).to(self.device)
        for i in range(bs):
            mask3d[i, :src_len[i], :src_len[i]] = 1
            mask2d[i, :src_len[i]] = 1

        if self.traffic_features:
            # 获取 rho
            rho = self.rho(src_rids)
            c, _, _ = self.c(data_S)
            logm, logv = self.traffic(rho, c)

            traffic_v = logm.permute(1, 0, 2).to(self.device)

            src = torch.cat((src, traffic_v), dim=2)


        src = self.fc_in(src)
        src = src.transpose(0, 1)

        outputs, g = self.transformer(src, speed_cons, g, mask3d, mask2d)
        g.ndata['lg'] = self.pred_out(g.ndata['x'])
        g.ndata['lg'] = mask_graph_log_softmax(g)

        # outputs = self.final_encoder(outputs, mask, norm=True)
        outputs = outputs.transpose(0, 1)  # [src len, bs, hid dim]

        # idx = [i for i in range(bs)]
        # hidden = outputs[[i - 1 for i in src_len], idx, :].unsqueeze(0)
        assert outputs.size(0) == max_src_len

        for i in range(bs):
            outputs[src_len[i]:, i, :] = 0
        hidden = torch.mean(outputs, dim=0).unsqueeze(0)

        if self.pro_features_flag:
            extra_emb = self.extra(pro_features)
            extra_emb = extra_emb.unsqueeze(0)
            # extra_emb = [1, batch size, extra output dim]
            hidden = torch.tanh(self.fc_hid(torch.cat((extra_emb, hidden), dim=2)))
            # hidden = [1, batch size, hid dim]

        return outputs, hidden, g


class Attention(nn.Module):
    """
        Calculate the attention score of the sequence with respect to the query vector.
        hidden: [1, batch size, hid dim] represents to query vector.
        encoder_outputs: [src len, batch size, hid dim * num directions] represents to key/value vectors.
        :return [batch size, src len] represents to attention score with sum of dim 1 to 1.
    """

    def __init__(self, parameters):
        super().__init__()
        self.hid_dim = parameters.hid_dim

        self.attn = nn.Linear(self.hid_dim * 2, self.hid_dim)
        self.v = nn.Linear(self.hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, attn_mask):
        # hidden = [1, bath size, hid dim]
        # encoder_outputs = [src len, batch size, hid dim * num directions]
        src_len = encoder_outputs.shape[0]
        # repeat decoder hidden sate src_len times
        hidden = hidden.repeat(src_len, 1, 1)
        hidden = hidden.permute(1, 0, 2)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # hidden = [batch size, src len, hid dim]
        # encoder_outputs = [batch size, src len, hid dim * num directions]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch size, src len, hid dim]

        attention = self.v(energy).squeeze(2)
        # attention = [batch size, src len]
        attention = attention.masked_fill(attn_mask == 0, -1e6)
        # using mask to force the attention to only be over non-padding elements.

        return F.softmax(attention, dim=1)


class DecoderMulti(nn.Module):
    """
        Trajectory Decoder.
        Set online_feature_flag=False.
        Keep tandem_fea_flag (road network static feature).

        Decoder: Attention + RNN
        If calculate attention, calculate the attention between current hidden vector and encoder output.
        Feed rid embedding, hidden vector, input rate into rnn to get the next prediction.
    """

    def __init__(self, parameters):
        super().__init__()

        self.id_size = parameters.id_size
        self.id_emb_dim = parameters.id_emb_dim
        self.hid_dim = parameters.hid_dim
        self.pro_output_dim = parameters.pro_output_dim
        self.online_dim = parameters.online_dim
        self.rid_fea_dim = parameters.rid_fea_dim

        self.attn_flag = parameters.attn_flag
        self.dis_prob_mask_flag = parameters.dis_prob_mask_flag  # final softmax
        self.online_features_flag = parameters.online_features_flag
        self.tandem_fea_flag = parameters.tandem_fea_flag

        rnn_input_dim = self.hid_dim + 1
        fc_id_out_input_dim = self.hid_dim
        fc_rate_out_input_dim = self.hid_dim

        type_input_dim = self.hid_dim + self.hid_dim
        self.tandem_fc = nn.Sequential(
            nn.Linear(type_input_dim, self.hid_dim),
            nn.ReLU()
        )

        if self.attn_flag:
            self.attn = Attention(parameters)
            rnn_input_dim = rnn_input_dim + self.hid_dim

        if self.online_features_flag:
            rnn_input_dim = rnn_input_dim + self.online_dim  # 5 poi and 5 road network

        if self.tandem_fea_flag:
            fc_rate_out_input_dim = self.hid_dim + self.rid_fea_dim

        self.rnn = nn.GRU(rnn_input_dim, self.hid_dim)
        self.fc_id_out = nn.Linear(fc_id_out_input_dim, self.id_size)
        self.fc_rate_out = nn.Linear(fc_rate_out_input_dim, 1)
        self.dropout = nn.Dropout(parameters.dropout)

    def forward(self, input_id, input_rate, hidden, encoder_outputs, attn_mask,
                constraint_vec, pro_features, online_features, rid_features):

        # input_id = [batch size, 1] rid long
        # input_rate = [batch size, 1] rate float.
        # hidden = [1, batch size, hid dim]
        # encoder_outputs = [src len, batch size, hid dim * num directions]
        # attn_mask = [batch size, src len]
        # constraint_vec = [batch size, id_size], [id_size] is the vector of reachable rid
        # pro_features = [batch size, profile features input dim]
        # online_features = [batch size, online features dim]
        # rid_features = [batch size, rid features dim]

        input_id = input_id.squeeze(1)  # cannot use squeeze() bug for batch size = 1
        # input_id = [batch size]
        input_rate = input_rate.unsqueeze(0)
        # input_rate = [1, batch size, 1]
        embedded = self.dropout(torch.index_select(self.emb_id, index=input_id, dim=0)).unsqueeze(0)
        # embedded = [1, batch size, emb dim]

        if self.attn_flag:
            a = self.attn(hidden, encoder_outputs, attn_mask)
            # a = [batch size, src len]
            a = a.unsqueeze(1)
            # a = [batch size, 1, src len]
            encoder_outputs = encoder_outputs.permute(1, 0, 2)
            # encoder_outputs = [batch size, src len, hid dim * num directions]
            weighted = torch.bmm(a, encoder_outputs)
            # weighted = [batch size, 1, hid dim * num directions]
            weighted = weighted.permute(1, 0, 2)
            # weighted = [1, batch size, hid dim * num directions]

            if self.online_features_flag:
                rnn_input = torch.cat((weighted, embedded, input_rate,
                                       online_features.unsqueeze(0)), dim=2)
            else:
                rnn_input = torch.cat((weighted, embedded, input_rate), dim=2)
        else:
            if self.online_features_flag:
                rnn_input = torch.cat((embedded, input_rate, online_features.unsqueeze(0)), dim=2)
            else:
                rnn_input = torch.cat((embedded, input_rate), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)

        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        if not (output == hidden).all():
            import pdb
            pdb.set_trace()
        assert (output == hidden).all()

        # pre_rid
        if self.dis_prob_mask_flag:
            prediction_id = mask_log_softmax(self.fc_id_out(output.squeeze(0)),
                                             constraint_vec, log_flag=True)
        else:
            prediction_id = F.log_softmax(self.fc_id_out(output.squeeze(0)), dim=1)
            # then the loss function should change to nll_loss()

        # pre_rate
        max_id = prediction_id.argmax(dim=1).long()
        id_emb = self.dropout(torch.index_select(self.emb_id, index=max_id, dim=0))
        rate_input = torch.cat((id_emb, hidden.squeeze(0)), dim=1)
        rate_input = self.tandem_fc(rate_input)  # [batch size, hid dim]
        if self.tandem_fea_flag:
            prediction_rate = torch.sigmoid(self.fc_rate_out(torch.cat((rate_input, rid_features), dim=1)))
        else:
            prediction_rate = torch.sigmoid(self.fc_rate_out(rate_input))

        # prediction_id = [batch size, id_size]
        # prediction_rate = [batch size, 1]

        return prediction_id, prediction_rate, hidden


class CustomGNN(nn.Module):
    def __init__(self, parameters):
        super(CustomGNN, self).__init__()
        self.gnn_type = parameters.gnn_type  # 可以选择不同的 GNN 类型
        self.node_input_dim = parameters.speed_dim  # 输入维度
        self.node_hidden_dim = parameters.speed_hid  # 隐藏层维度
        self.num_layers = parameters.num_layers  # GNN 的层数
        self.dropout_rate = parameters.dropout  # dropout 的概率

        # 根据选择的 gnn_type 初始化不同的 GNN 模型
        if self.gnn_type == 'gat':
            self.gnn = UnsupervisedGAT(self.node_input_dim, self.node_hidden_dim, edge_input_dim=0,
                                       num_layers=self.num_layers)
        elif self.gnn_type == 'gcn':
            self.gnn = UnsupervisedGCN(self.node_input_dim, self.node_hidden_dim, num_layers=self.num_layers)
        else:
            self.gnn = UnsupervisedGIN(self.node_input_dim, self.node_hidden_dim, edge_input_dim=0,
                                       num_layers=self.num_layers)

        # dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, g, x, readout=True):
        '''
        :param g: DGLGraph 输入的图
        :param x: 节点的特征嵌入，大小为 [node size, input_dim]
        :param readout: 是否进行全图池化，默认为 True
        :return: 如果 readout=True，返回 [graph size, hidden dim]，否则返回 [node size, hidden dim]
        '''
        mask = (x.sum(dim=1) != 0).float()
        x = x * mask.unsqueeze(1)
        x = self.dropout(self.gnn(g, x))
        if not readout:
            return x
        g.ndata['v'] = x

        if 'w' in g.ndata:
            return dgl.mean_nodes(g, 'v', weight='w'), g
        else:
            return dgl.mean_nodes(g, 'v'), g
        
class ProbSpeed(nn.Module):
    def __init__(self, n_in, hidden_size, dim_s, dropout=0.2, use_selu=True):
        super(ProbSpeed, self).__init__()

        self.fc1 = nn.Linear(n_in, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, dim_s)
        self.fc_logvar = nn.Linear(hidden_size, dim_s)
        self.use_selu = use_selu
        self.dropout = nn.Dropout(dropout)

    def reparameterize(self, mu, logvar):
        """ 重参数化技巧：从高斯分布中采样"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        """
        :param x: 输入的速度特征，[batch_size, n_in]
        :return: (speed_emb, mu, logvar)
        """

        h = self.fc1(x)
        if self.use_selu:
            h = F.selu(h)
        else:
            h = F.leaky_relu(h, negative_slope=0.1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)  #
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar



class Seq2SeqMulti(nn.Module):
    """
    Trajectory Seq2Seq Model.
    """

    def __init__(self, encoder, decoder, device, parameters):
        super().__init__()
        self.id_size = parameters.id_size
        self.hid_dim = parameters.hid_dim
        self.grid_num = parameters.grid_num
        self.id_emb_dim = parameters.id_emb_dim
        self.grid_flag = parameters.grid_flag
        self.traffic_flag = parameters.traffic_features
        self.speed_flag = parameters.speed_features
        
        self.dis_prob_mask_flag = parameters.dis_prob_mask_flag
        self.subg = parameters.subg
        self.emb_id = nn.Parameter(torch.rand(self.id_size, self.id_emb_dim))
        self.device = device
        self.grid_id = nn.Parameter(torch.rand(self.grid_num[0], self.grid_num[1], self.id_emb_dim))
        self.rn_grid_dict = parameters.rn_grid_dict
        self.pad_rn_grid, _ = self.merge(self.rn_grid_dict)
        # self.grid_len = [fea.shape[0] - 1 for fea in self.rn_grid_dict]
        self.grid_len = torch.tensor([fea.shape[0] for fea in self.rn_grid_dict])

        self.gnn = RoadGNN(parameters)
        self.grid = nn.GRU(self.id_emb_dim, self.id_emb_dim)
        self.encoder = encoder  # Encoder
        self.decoder = decoder  # DecoderMulti

        self.params = parameters

        self.speed_dim = parameters.speed_dim
        self.speed_hid = parameters.speed_hid
        self.speed_in = nn.Linear(5, self.speed_dim).to(self.device)
        self.speedgnn = CustomGNN(parameters)
        
        self.noise_flag = parameters.use_noise
        if self.noise_flag:
            self.prob_speed = ProbSpeed(n_in=self.speed_dim, hidden_size=parameters.hidden_size2, dim_s=self.speed_hid)

    def merge(self, sequences):
        lengths = [len(seq) for seq in sequences]
        dim = sequences[0].size(1)  # get dim for each sequence
        padded_seqs = torch.zeros(len(sequences), max(lengths), dim)

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths


    def forward(self, src, src_len, src_rids, speed, data_S, trg_id, trg_rate, trg_len,
                constraint_mat_trg, pro_features,
                online_features_dict, rid_features_dict, constraint_graph_src,
                src_gps_seqs, teacher_forcing_ratio=0.5):
        """
        src = [src len, batch size, 3], x,y,t
        src_len = [batch size]
        trg_id = [trg len, batch size, 1]
        trg_rate = [trg len, batch size, 1]
        trg_len = [batch size]
        constraint_mat = [trg len, batch size, id_size]
        pro_features = [batch size, profile features input dim]
        online_features_dict = {rid: online_features} # rid --> grid --> online features
        rid_features_dict = {rid: rn_features}
        constraint_src = [src len, batch size, id size]
        teacher_forcing_ratio is probability to use teacher forcing
        e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        Return:
        ------
        outputs_id: [seq len, batch size, id_size(1)] based on beam search
        outputs_rate: [seq len, batch size, 1]
        """
        max_trg_len = trg_id.size(0)
        max_src_len = src.size(0)
        batch_size = trg_id.size(1)

        # road representation
        max_grid_len = self.pad_rn_grid.size(1)
        rn_grid = self.pad_rn_grid.reshape(-1, 2)
        grid_input = self.grid_id[rn_grid.numpy()[:, 0], rn_grid.numpy()[:, 1], :]
        grid_input = grid_input.reshape(self.id_size, max_grid_len, -1).transpose(0, 1)

        # change to pad_packed_sequence
        packed_grid_input = nn.utils.rnn.pack_padded_sequence(grid_input, self.grid_len,
                                                              batch_first=False, enforce_sorted=False)
        _, grid_output = self.grid(packed_grid_input)
        grid_emb = grid_output.reshape(-1, self.id_emb_dim)
        assert grid_emb.size(0) == self.emb_id.size(0)
        # grid_emb = grid_output[self.grid_len, range(len(self.grid_len)), :]  # [rid, dim]

        input_road = torch.index_select(self.emb_id, index=self.subg.ndata['id'].long(), dim=0)
        input_grid = torch.index_select(grid_emb, index=self.subg.ndata['id'].long(), dim=0)
        input_emb = F.leaky_relu(input_road + input_grid)
        # input_emb = torch.cat((input_road, input_grid), dim=-1)
        # finish changing

        road_emb, _ = self.gnn(self.subg, input_emb)
        road_emb = road_emb.reshape(-1, self.hid_dim)
        self.decoder.emb_id = road_emb  # [id size, hidden dim]

        # road speed
        if self.speed_flag:
            speed = speed.to(self.device).float()
            speed_emb = self.speed_in(speed)
            if self.noise_flag:
                speed_emb, mu, logvar = self.prob_speed(speed_emb)
            input_speed = torch.index_select(speed_emb, index=self.subg.ndata['id'].long(), dim=0)
            input_speed = torch.index_select(speed_emb, index=self.subg.ndata['id'].long(), dim=0)
            speed_emb, _ = self.speedgnn(self.subg, input_speed)
            speed_emb = speed_emb.reshape(-1, self.speed_hid)
        else:
            speed_emb = None

        assert self.dis_prob_mask_flag
        input_cons = torch.index_select(road_emb, index=constraint_graph_src.ndata['id'].long(),
                                        dim=0)

        constraint_graph_src.ndata['x'] = input_cons
        cons_emb = dgl.mean_nodes(constraint_graph_src, 'x', weight='w')
        cons_emb = cons_emb.reshape(batch_size, max_src_len, -1).transpose(0, 1)
        if self.speed_flag:
            speed_cons = torch.index_select(speed_emb, index=constraint_graph_src.ndata['id'].long(),
                                            dim=0)
            constraint_graph_src.ndata['v'] = speed_cons
            # cons_speed_emb = dgl.mean_nodes(constraint_graph_src, 'v', weight='w')
            # speed_src = cons_speed_emb.reshape(batch_size, max_src_len, -1).transpose(0, 1)
        else:
            speed_cons  = None

        if self.grid_flag:
            grid_input = src[:, :, :2].reshape(-1, 2).cpu().numpy()
            grid_emb = self.grid_id[grid_input[:, 0].tolist(), grid_input[:, 1].tolist(), :]
            grid_emb = grid_emb.reshape(max_src_len, batch_size, -1)
            src = torch.cat((cons_emb, grid_emb, src), dim=-1)
        else:
            src = torch.cat((cons_emb, src), dim=-1)
        
        # if self.traffic_flag:
        #     src = torch.cat((traffic_emb, src), dim=-1)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hiddens, g = self.encoder(src, speed_cons, src_len, src_rids, data_S, constraint_graph_src, pro_features)

        if self.decoder.attn_flag:
            attn_mask = torch.zeros(batch_size, max(src_len))  # only attend on unpadded sequence
            for i in range(len(src_len)):
                attn_mask[i][:src_len[i]] = 1.
            attn_mask = attn_mask.to(self.device)
        else:
            attn_mask = None

        outputs_id, outputs_rate = self.normal_step(max_trg_len, batch_size, trg_id, trg_rate, trg_len,
                                                    encoder_outputs, hiddens, attn_mask,
                                                    online_features_dict,
                                                    rid_features_dict,
                                                    constraint_mat_trg, pro_features,
                                                    teacher_forcing_ratio)

        return outputs_id, outputs_rate, g

    def normal_step(self, max_trg_len, batch_size, trg_id, trg_rate, trg_len, encoder_outputs, hidden,
                    attn_mask, online_features_dict, rid_features_dict,
                    constraint_mat, pro_features, teacher_forcing_ratio):
        """
        Returns:
        -------
        outputs_id: [seq len, batch size, id size]
        outputs_rate: [seq len, batch size, 1]
        """
        # tensor to store decoder outputs
        outputs_id = torch.zeros(max_trg_len, batch_size, self.decoder.id_size).to(self.device)
        outputs_rate = torch.zeros(trg_rate.size()).to(self.device)

        # first input to the decoder is the <sos> tokens
        input_id = trg_id[0, :]
        input_rate = trg_rate[0, :]
        for t in range(1, max_trg_len):
            # insert input token embedding, previous hidden state, all encoder hidden states
            #  and attn_mask
            # receive output tensor (predictions) and new hidden state
            if self.decoder.online_features_flag:
                online_features = get_dict_info_batch(input_id, online_features_dict).to(self.device)
            else:
                online_features = torch.zeros((1, batch_size, self.decoder.online_dim))
            if self.decoder.tandem_fea_flag:
                rid_features = get_dict_info_batch(input_id, rid_features_dict).to(self.device)
            else:
                rid_features = None
            prediction_id, prediction_rate, hidden = self.decoder(input_id, input_rate, hidden, encoder_outputs,
                                                                  attn_mask, constraint_mat[t], pro_features,
                                                                  online_features, rid_features)

            # place predictions in a tensor holding predictions for each token
            outputs_id[t] = prediction_id
            outputs_rate[t] = prediction_rate

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1_id = prediction_id.argmax(1)
            top1_id = top1_id.unsqueeze(-1)  # make sure the output has the same dimension as input

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input_id = trg_id[t] if teacher_force else top1_id
            input_rate = trg_rate[t] if teacher_force else prediction_rate

        # max_trg_len, batch_size, trg_rid_size
        outputs_id = outputs_id.permute(1, 0, 2)  # batch size, seq len, rid size
        outputs_rate = outputs_rate.permute(1, 0, 2)  # batch size, seq len, 1

        for i in range(batch_size):
            outputs_id[i][trg_len[i]:] = -100
            outputs_id[i][trg_len[i]:, 0] = 0  # make sure argmax will return eid0
            outputs_rate[i][trg_len[i]:] = 0
        outputs_id = outputs_id.permute(1, 0, 2)
        outputs_rate = outputs_rate.permute(1, 0, 2)

        return outputs_id, outputs_rate
