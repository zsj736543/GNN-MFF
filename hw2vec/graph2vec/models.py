
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv, SAGPooling
from torch_geometric.nn import global_max_pool

class GATNet(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GATNet, self).__init__()
        num_heads = 8
        self.conv1 = GATConv(num_features, 20, heads=num_heads)
        self.conv2 = GATConv(20 * num_heads, num_classes, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

class ast_block(nn.Module):
    def __init__(self, config, data_proc2):
        super(ast_block, self).__init__()
        self.config = config
        self.conv1 = GATNet(data_proc2.num_node_labels, config.hidden)
        self.conv4 = GATNet(config.hidden, config.hidden)

        self.pool = SAGPooling(config.hidden, config.poolratio)

    def forward(self, x, edge_index, batch):
        attn_weights = dict()
        x = F.one_hot(x, num_classes=self.config.num_feature_dim).float()
        x = F.dropout(F.relu(self.conv1(x, edge_index)), p=self.config.dropout, training=self.training)
        x = F.dropout(F.relu(self.conv4(x, edge_index)), p=self.config.dropout, training=self.training)
        x, edge_index, _, batch, attn_weights['pool_perm'], attn_weights['pool_score'] = \
            self.pool(x, edge_index, batch=batch)
        x = global_max_pool(x, batch)

        attn_weights['batch'] = batch
        return x, attn_weights  #这里的attn_weights是否代表了权重

# 6、 DFG网络搭建
class dfg_block(nn.Module):
    def __init__(self, config, data_proc1):
        super(dfg_block, self).__init__()
        self.config = config
        self.conv1 = GCNConv(79, config.hidden)
        self.conv2 = GCNConv(config.hidden, config.hidden)

        self.pool = SAGPooling(config.hidden, config.poolratio)

    def forward(self, x, edge_index, batch):
        attn_weights = dict()
        x = F.one_hot(x, num_classes=self.config.num_feature_dim).float()
        x = F.dropout(F.relu(self.conv1(x, edge_index)), p=self.config.dropout, training=self.training)
        x = F.dropout(F.relu(self.conv2(x, edge_index)), p=self.config.dropout, training=self.training)
        x, edge_index, _, batch, attn_weights['pool_perm'], attn_weights['pool_score'] = \
            self.pool(x, edge_index, batch=batch)
        x = global_max_pool(x, batch)
        attn_weights['batch'] = batch
        return x, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.linear_query = nn.Linear(input_size, hidden_size)
        self.linear_key = nn.Linear(input_size, hidden_size)
        self.linear_value = nn.Linear(input_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, hidden_size)

    def forward(self, input1, input2):

        query = self.linear_query(input1)
        key = self.linear_key(input2)
        value = self.linear_value(input2)

        query = query.view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)


        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights.transpose(-2, -1), value)

        output = attention_output.transpose(1, 2).contiguous().view(1, 240, -1)

        return output

class AFFGCN(nn.Module):
    def __init__(self, num_wx_features, num_for_predict, num_nodes, cfg1, cfg2, data_proc1, data_proc2):
        super(AFFGCN, self).__init__()
        self.block_tree = ast_block(cfg2, data_proc2)

        self.block_graph = dfg_block(cfg1, data_proc1)

        self.out_attention = MultiHeadAttention(input_size=cfg1.hidden, hidden_size=cfg1.hidden, num_heads=8)

    def forward(self, x_tree, edge_index_tree, x_graph, edge_index_graph, batch2, batch1):
        x_tf, tree_weights = self.block_tree(x_tree, edge_index_tree, batch2)

        out_wx, graph_weights = self.block_graph(x_graph, edge_index_graph, batch1)

        out = self.out_attention(out_wx, x_tf)

        return out, out_wx

    def mlp(self, x):
        x = x.view(1, 240)
        self.fc = nn.Linear(240,2).to('cuda:0')
        return self.fc(x)




