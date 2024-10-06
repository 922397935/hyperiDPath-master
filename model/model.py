import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from base import BaseModel


# 超图卷积核
class HyperGraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True, use_bn=False, drop_rate=0.5, is_last=False):
        super(HyperGraphConvolution, self).__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_features) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_features, out_features, bias=bias)

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, node_input, hyperedge_input, node_lap, hyperedge_lap):
        r"""
            args: node_input: (17895, 64)
                hyperedge_input(396656, 64)
                node_lap: (17895, 17895)
                hyperedge_lap(396656, 396656)
            第二次进来时发生报错：
            self.weight: 128, 64
            node_input: 414550 128
        """
        # node_input = self.theta(node_input)
        # hyperedge_input = self.theta(hyperedge_input)

        support_1 = torch.mm(node_input, self.weight)  #
        output_1 = torch.spmm(node_lap, support_1)
        support_2 = torch.mm(hyperedge_input, self.weight)
        output_2 = torch.spmm(hyperedge_lap, support_2)
        # output_1 17895, 128, output_2 396656, 128
        # output = torch.cat((output_1, output_2), dim=0) 
        # print(1)
        if not self.is_last:
            output_1 = self.act(output_1)
            output_2 = self.act(output_2)
            if self.bn is not None:
                output_1 = self.bn(output_1)
                output_2 = self.bn(output_2)
            output_1 = self.drop(output_1)
            output_2 = self.drop(output_2)

        if self.bias is not None:
            return output_1 + self.bias, output_2 + self.bias
        else:
            return output_1, output_2

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

# 超图卷积层模型
class HyperGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, use_bn: False):
        super(HyperGCN, self).__init__()
        self.hypergc1 = HyperGraphConvolution(nfeat, nhid, use_bn=use_bn,drop_rate=dropout)
        self.hypergc2 = HyperGraphConvolution(nhid, nclass, use_bn=use_bn, drop_rate=dropout, is_last=True)
        self.dropout = dropout

    def forward(self, x, y, lap_x, lap_y):
        x, y = self.hypergc1(x, y, lap_x, lap_y)  # x (414550, 128)
        # TODO这段需要加吗？
        x = F.relu(x)
        y = F.relu(y)
        x = F.dropout(x, self.dropout, training=self.training)
        y = F.dropout(y, self.dropout, training=self.training)
        
        # 这种方式造成的问题就是这里传入的x的值变了
        x, y = self.hypergc2(x, y, lap_x, lap_y) 
        result = torch.cat((x, y), dim=0)
        # 这里的return 要不要合并，方便path_feature嵌入？
        # return F.log_softmax(x, dim=1), F.log_softmax(y, dim=1) 
        return F.log_softmax(result, dim=1)


class hyperiDPath(BaseModel):
    def __init__(self, node_num, hyperedge_num, type_num, node_lap, hyperedge_lap, emb_dim=16, hypergcn_layersize=[16, 16, 16], dropout=0.5):
        # 初始化时我需要传入hyperedge吗？
        super().__init__()
        self.node_num = node_num
        self.hyperedge_num = hyperedge_num
        self.type_num = type_num
        self.node_lap = node_lap
        self.hyperedge_lap = hyperedge_lap
        self.emb_dim = emb_dim
        self.node_embedding = nn.Embedding(node_num + 1, emb_dim, padding_idx=0)  # 表示的是个数的嵌入
        self.hyperedge_embedding = nn.Embedding(hyperedge_num + 1, emb_dim, padding_idx=0)  # 表示的个数嵌入
        self.type_embedding = nn.Embedding(type_num + 1, emb_dim, padding_idx=0)
        self.hypergcn = HyperGCN(nfeat=hypergcn_layersize[0], nhid=hypergcn_layersize[1],
                             nclass=hypergcn_layersize[2], dropout=dropout, use_bn=True)
        self.lstm = nn.LSTM(input_size=emb_dim * 2, hidden_size=emb_dim)

        self.node_attention_linear = nn.Linear(in_features=emb_dim, out_features=1, bias=False)
        self.node_attention_softmax = nn.Softmax(dim=1)

        # 这个直接写下来感觉并没有区分开
        self.hyperedge_attention_linear = nn.Linear(in_features=emb_dim, out_features=1, bias=False)
        self.hyperedge_attention_softmax = nn.Softmax(dim=1)

        self.path_attention_linear = nn.Linear(in_features=emb_dim, out_features=1, bias=False)
        self.path_attention_softmax = nn.Softmax(dim=1)

        self.output_linear = nn.Linear(in_features=emb_dim, out_features=1)

    def forward(self, path_feature, type_feature, lengths, mask, hypergcn=True):
        # shape of path_feature: [batch_size, path_num, path_length]
        # shape of type_feature: [batch_size, path_num, path_length]
        '''HGNN embedding'''
        total_node = torch.LongTensor([list(range(self.node_num + 1))]).to(path_feature.device)
        # IndexError: index out of range in self
        # 这里该不该加1？加1之后414551
        total_hyperedge = torch.LongTensor([list(range(self.hyperedge_num + 1))]).to(path_feature.device)
        
        # ego_value_embedding: 怎么写？
        node_value_embedding = self.node_embedding(total_node).squeeze()  # 17895, 64
        hyperedge_value_embedding = self.hyperedge_embedding(total_hyperedge).squeeze()  # 396650, 64

        total_value_embedding = torch.cat((node_value_embedding, hyperedge_value_embedding), dim=0)  # 414550, 64
        if hypergcn:
            hypergcn_value_embedding = self.hypergcn(x=node_value_embedding, y = hyperedge_value_embedding, lap_x=self.node_lap.to(path_feature.device),
                                                     lap_y=self.hyperedge_lap.to(path_feature.device))
        else:
            hypergcn_value_embedding = total_value_embedding  # 这个hypergcn为false，返不回4个值了吧

        '''Embedding'''
        batch, path_num, path_len = path_feature.size()
        path_feature = path_feature.view(batch * path_num, path_len)
        # shape of path_embedding: [batch_size*path_num, path_length, emb_dim]
        # 这里path_feature能分成两段吗？还是合一？
        path_embedding = hypergcn_value_embedding[path_feature]
        type_feature = type_feature.view(batch * path_num, path_len)
        # shape of type_embedding: [batch_size*path_num, path_length, emb_dim]
        # 多加个虚拟节点，type_embedding会不会报错？
        type_embedding = self.type_embedding(type_feature).squeeze()
        # shape of feature: [batch_size*path_num, path_length, emb_dim]
        feature = torch.cat((path_embedding, type_embedding), 2)

        '''Pack padded sequence'''
        feature = torch.transpose(feature, dim0=0, dim1=1)
        feature = utils.rnn.pack_padded_sequence(feature, lengths=list(lengths.view(batch * path_num).data),
                                                 enforce_sorted=False)

        '''LSTM'''
        # shape of lstm_out: [path_length, batch_size*path_num, emb_dim]
        lstm_out, _ = self.lstm(feature)
        # unpack, shape of lstm_out: [batch_size*path_num, path_length, emb_dim]
        lstm_out, _ = utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=path_len)

        '''Node attention'''
        # shape of output_path_embedding: [batch_size*path_num, emb_dim]
        mask = mask.view(batch * path_num, path_len)
        output_path_embedding, node_weight_normalized = self.node_attention(lstm_out, mask)
        # the original shape of node_weight_normalized: [batch_size*path_num, path_length]
        node_weight_normalized = node_weight_normalized.view(batch, path_num, path_len)
        # shape of output_path_embedding: [batch_size, path_num, emb_dim]
        # 这是节点路径注意力嵌入的输出，是该跟下面的合并？还是不运行？
        output_path_embedding = output_path_embedding.view(batch, path_num, self.emb_dim)

        '''Hyperedge attention'''
        output_path_embedding, hyperedge_weight_normalized = self.hyperedge_attention(lstm_out, mask)
      
        hyperedge_weight_normalized = hyperedge_weight_normalized.view(batch, path_num, path_len)
        
        output_path_embedding = output_path_embedding.view(batch, path_num, self.emb_dim)

        '''Path attention'''
       
        output_embedding, path_weight_normalized = self.path_attention(output_path_embedding)

        '''Prediction'''
        output = self.output_linear(output_embedding)

        return output, node_weight_normalized, hyperedge_weight_normalized, path_weight_normalized, 

    def node_attention(self, input, mask):
        # the shape of input: [batch_size*path_num, path_length, emb_dim]
        weight = self.node_attention_linear(input)  # shape: [batch_size*path_num, path_length, 1]
        # shape: [batch_size*path_num, path_length]
        weight = weight.squeeze()
        '''mask'''
        # the shape of mask: [batch_size*path_num, path_length]
        weight = weight.masked_fill(mask == 0, torch.tensor(-1e9))
        # shape: [batch_size*path_num, path_length]
        weight_normalized = self.node_attention_softmax(weight)
        # shape: [batch_size*path_num, path_length, 1]
        weight_expand = torch.unsqueeze(weight_normalized, dim=2)
        # shape: [batch_size*path_num, emb_dim]
        input_weighted = (input * weight_expand).sum(dim=1)
        return input_weighted, weight_normalized

    def hyperedge_attention(self, input, mask):
        # the shape of input: [batch_size*path_num, path_length, emb_dim]
        weight = self.hyperedge_attention_linear(input)  # shape: [batch_size*path_num, path_length, 1]
        # shape: [batch_size*path_num, path_length]
        weight = weight.squeeze()
        '''mask'''
        # the shape of mask: [batch_size*path_num, path_length]
        weight = weight.masked_fill(mask == 0, torch.tensor(-1e9))
        # shape: [batch_size*path_num, path_length]
        weight_normalized = self.node_attention_softmax(weight)
        # shape: [batch_size*path_num, path_length, 1]
        weight_expand = torch.unsqueeze(weight_normalized, dim=2)
        # shape: [batch_size*path_num, emb_dim]
        input_weighted = (input * weight_expand).sum(dim=1)
        return input_weighted, weight_normalized

    def path_attention(self, input):
        # the shape of input: [batch_size, path_num, emb_dim]
        weight = self.path_attention_linear(input)
        # [batch_size, path_num]
        weight = weight.squeeze()
        # [batch_size, path_num]
        weight_normalized = self.path_attention_softmax(weight)
        # [batch_size, path_num, 1]
        weight_expand = torch.unsqueeze(weight_normalized, dim=2)
        # [batch_size, emb_dim]
        input_weighted = (input * weight_expand).sum(dim=1)
        return input_weighted, weight_normalized