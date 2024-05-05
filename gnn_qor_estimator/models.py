import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import JumpingKnowledge, TransformerConv, GraphConv
from torch_geometric.nn import global_add_pool
import torch.nn as nn
import torch_geometric.nn as geonn
from scipy.stats import rankdata, kendalltau

from torch.nn import Sequential, Linear, ReLU


from os.path import join

from collections import OrderedDict, defaultdict


from torch_scatter import scatter_add
from torch_geometric.utils import softmax

from torch_geometric.nn.inits import reset


# This file uses code from GNN-DSE (https://github.com/UCLA-VAST/GNN-DSE), under the BSD 3-Clause License. Copyright (c) 2022, UCLA VAST Lab. All rights reserved. 
    
# from GNN-DSE https://github.com/UCLA-VAST/GNN-DSE
class GNNDSE(torch.nn.Module):
    def __init__(self, num_layers, in_channels, edge_dim = 7, res=False, attention_pool=True,):
        super(GNNDSE, self).__init__()
        
        conv_class = TransformerConv

        self.attention_pool = attention_pool

        #dimensions of inner layers
        D = 64

        # print(in_channels)
        self.conv_first = conv_class(in_channels, D, edge_dim=edge_dim)

        self.conv_layers = nn.ModuleList()

        for _ in range(num_layers - 1):
            conv = conv_class(D, D, edge_dim=edge_dim)
 
            self.conv_layers.append(conv)

        self.jkn = JumpingKnowledge('max', channels=D, num_layers=2)


        self.out_dim = 1
        self.loss_function = torch.nn.MSELoss()       
        
        self.gate_nn = Sequential(Linear(D, D), ReLU(), Linear(D, 1))
        self.glob = MyGlobalAttention(self.gate_nn, None)

        self.MLPs = nn.ModuleList()
        
        if D > 64:
            hidden_channels = [D // 2, D // 4, D // 8, D // 16, D // 32]
        else:
            hidden_channels = [D // 2, D // 4, D // 8]

        for i in range(6):
            mlp = MLP(D, self.out_dim, activation_type="elu",
                                    hidden_channels=hidden_channels,
                                    num_hidden_lyr=len(hidden_channels))
                            
            self.MLPs.append(mlp)


    def forward(self, data):
        x, edge_index, edge_attr, batch= \
            data.x, data.edge_index, data.edge_attr, data.batch# , data.pragmas

        activation = F.elu

        outs = []

        out = activation(self.conv_first(x, edge_index, edge_attr=edge_attr))
        outs.append(out)

        for i, conv in enumerate(self.conv_layers):
            out = conv(out, edge_index, edge_attr=edge_attr)
            if i != len(self.conv_layers) - 1:
                out = activation(out)
                
            outs.append(out)

        if self.jkn:
            out = self.jkn(outs)
        
        if self.attention_pool:
            out, _ = self.glob(out, batch)
        else:
            out = global_add_pool(out, batch)


        out_dict = OrderedDict()
        total_loss = 0
        out_embed = out
        
        loss_dict = {}
        for i, target in enumerate(["LUTs", "FFs", "BRAMs", "DSPs", "clock", "latency"]):
            ground_truth = data.y[:, i].reshape([-1, 1])
            out = self.MLPs[i](out_embed)

            loss = torch.sqrt(self.loss_function(out, ground_truth))
                
            out_dict[target] = out
            total_loss += loss
            loss_dict[target] = loss


        return out_dict, total_loss, loss_dict
    

class Hier(torch.nn.Module):
    def __init__(self, in_channels, edge_dim = 7):
        super(Hier, self).__init__()
        
        conv_class = TransformerConv

        #dimensions of inner layers
        D = 64

        # print(in_channels)
        self.conv_first = conv_class(in_channels, D, edge_dim=edge_dim)

        self.conv_2 = conv_class(D, D, edge_dim=edge_dim)      

        self.out_dim = 1
        self.loss_function = torch.nn.MSELoss()
        
        
        self.gate_nn = Sequential(Linear(D, D), ReLU(), Linear(D, 1))
        self.glob = MyGlobalAttention(self.gate_nn, None)

        self.bb_conv = conv_class(D, D)

        self.bb_gate_nn = Sequential(Linear(D, D), ReLU(), Linear(D, 1))
        self.bb_glob = MyGlobalAttention(self.bb_gate_nn, None)

        self.resBlock1 = ResBlock(in_channels)
        self.resBlock2 = ResBlock(D)
        self.resBlock3 = ResBlock(D)
        self.resBlock4 = ResBlock(D)

        self.MLPs = nn.ModuleList()
        
        if D > 64:
            hidden_channels = [D // 2, D // 4, D // 8, D // 16, D // 32]
        else:
            hidden_channels = [D // 2, D // 4, D // 8]

        for i in range(6):
            mlp = MLP(D, self.out_dim, activation_type="elu",
                                    hidden_channels=hidden_channels,
                                    num_hidden_lyr=len(hidden_channels))
                            
            self.MLPs.append(mlp)
    

    def forward(self, data):
        x, edge_index, edge_attr, batch= \
            data.x, data.edge_index, data.edge_attr, data.batch# , data.pragmas

        activation = F.elu

        outs = []

        out =   self.resBlock1(x)

        out = activation(self.conv_first(out, edge_index, edge_attr=edge_attr))

        out = self.resBlock2(out)

        out = self.conv_2(out, edge_index, edge_attr=edge_attr)

        out, _ = self.glob(out, data.cfg_select)
        out = self.resBlock3(out)


        out = self.bb_conv(out,data.cfg_edge_index)

        out, _ = self.bb_glob(out, data.bb_batch)

        out = self.resBlock4(out)

        out_dict = OrderedDict()
        total_loss = 0
        out_embed = out
        
        loss_dict = {}
        for i, target in enumerate(["LUTs", "FFs", "BRAMs", "DSPs", "clock", "latency"]):
            ground_truth = data.y[:, i].reshape([-1, 1])
            out = self.MLPs[i](out_embed)

            loss = torch.sqrt(self.loss_function(out, ground_truth))
                
            out_dict[target] = out
            total_loss += loss
            loss_dict[target] = loss


        return out_dict, total_loss, loss_dict


        
class ArchC(torch.nn.Module):
    def __init__(self, in_channels, edge_dim = 7):
        super(ArchC, self).__init__()
        
        conv_class = TransformerConv

        #dimensions of inner layers
        D = 64

        # print(in_channels)
        self.conv_first = conv_class(in_channels, D, edge_dim=edge_dim)

        self.conv_2 = conv_class(D, D, edge_dim=edge_dim)

        self.out_dim = 1
        self.loss_function = torch.nn.MSELoss()
        
        
        self.gate_nn = Sequential(Linear(D, D), ReLU(), Linear(D, 1))
        self.glob = MyGlobalAttention(self.gate_nn, None)


        self.resBlock1 = ResBlock(in_channels)
        self.resBlock2 = ResBlock(D)
        self.resBlock3 = ResBlock(D)

        self.MLPs = nn.ModuleList()
        
        if D > 64:
            hidden_channels = [D // 2, D // 4, D // 8, D // 16, D // 32]
        else:
            hidden_channels = [D // 2, D // 4, D // 8]

        for i in range(6):
            mlp = MLP(D, self.out_dim, activation_type="elu",
                                    hidden_channels=hidden_channels,
                                    num_hidden_lyr=len(hidden_channels))
                            
            self.MLPs.append(mlp)


    def forward(self, data):
        x, edge_index, edge_attr, batch= \
            data.x, data.edge_index, data.edge_attr, data.batch# , data.pragmas

        activation = F.elu

        outs = []

        out =   self.resBlock1(x)

        out = activation(self.conv_first(out, edge_index, edge_attr=edge_attr))

        out = self.resBlock2(out)

        out = self.conv_2(out, edge_index, edge_attr=edge_attr)


        out, _ = self.glob(out, batch)

        out = self.resBlock3(out)


        out_dict = OrderedDict()
        total_loss = 0
        out_embed = out
        
        loss_dict = {}
        for i, target in enumerate(["LUTs", "FFs", "BRAMs", "DSPs", "clock", "latency"]):
            ground_truth = data.y[:, i].reshape([-1, 1])
            out = self.MLPs[i](out_embed)

            loss = torch.sqrt(self.loss_function(out, ground_truth))
                
            out_dict[target] = out
            total_loss += loss
            loss_dict[target] = loss


        return out_dict, total_loss, loss_dict

class ResBlock(nn.Module):
    def __init__(self, num_features):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Linear(num_features, num_features)
        self.bn1 = nn.BatchNorm1d(num_features)
        self.conv2 = nn.Linear(num_features, num_features)
        self.bn2 = nn.BatchNorm1d(num_features)


    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

# from GNN-DSE https://github.com/UCLA-VAST/GNN-DSE
class MyGlobalAttention(torch.nn.Module):
    r"""Global soft attention layer from the `"Gated Graph Sequence Neural
    Networks" <https://arxiv.org/abs/1511.05493>`_ paper

    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathrm{softmax} \left(
        h_{\mathrm{gate}} ( \mathbf{x}_n ) \right) \odot
        h_{\mathbf{\Theta}} ( \mathbf{x}_n ),

    where :math:`h_{\mathrm{gate}} \colon \mathbb{R}^F \to
    \mathbb{R}` and :math:`h_{\mathbf{\Theta}}` denote neural networks, *i.e.*
    MLPS.

    Args:
        gate_nn (torch.nn.Module): A neural network :math:`h_{\mathrm{gate}}`
            that computes attention scores by mapping node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, 1]`, *e.g.*,
            defined by :class:`torch.nn.Sequential`.
        nn (torch.nn.Module, optional): A neural network
            :math:`h_{\mathbf{\Theta}}` that maps node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, out_channels]`
            before combining them with the attention scores, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
    """
    def __init__(self, gate_nn, nn=None):
        super(MyGlobalAttention, self).__init__()
        self.gate_nn = gate_nn
        self.nn = nn

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)
    
    def forward(self, x, batch, size=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        gate = self.gate_nn(x).view(-1, 1)
        
        x = self.nn(x) if self.nn is not None else x

        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = softmax(gate, batch)
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)

        return out, gate

    def __repr__(self):
        return '{}(gate_nn={}, nn={})'.format(self.__class__.__name__,
                                              self.gate_nn, self.nn)
    
# from GNN-DSE https://github.com/UCLA-VAST/GNN-DSE
class MLP(nn.Module):
    '''mlp can specify number of hidden layers and hidden layer channels'''

    def __init__(self, input_dim, output_dim, activation_type='relu', num_hidden_lyr=2,
                 hidden_channels=None, bn=False):
        super().__init__()
        self.out_dim = output_dim
        if not hidden_channels:
            hidden_channels = [input_dim for _ in range(num_hidden_lyr)]
        elif len(hidden_channels) != num_hidden_lyr:
            raise ValueError(
                "number of hidden layers should be the same as the lengh of hidden_channels")
        self.layer_channels = [input_dim] + hidden_channels + [output_dim]
        self.activation = create_act(activation_type)
        self.layers = nn.ModuleList(list(
            map(self.weight_init, [nn.Linear(self.layer_channels[i], self.layer_channels[i + 1])
                                   for i in range(len(self.layer_channels) - 1)])))
        self.bn = bn
        if self.bn:
            self.bn = torch.nn.BatchNorm1d(output_dim)

    def weight_init(self, m):
        torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        return m

    def forward(self, x):
        layer_inputs = [x]
        for layer in self.layers:
            input = layer_inputs[-1]
            if layer == self.layers[-1]:
                layer_inputs.append(layer(input))
            else:
                layer_inputs.append(self.activation(layer(input)))
        # model.store_layer_output(self, layer_inputs[-1])
        if self.bn:
            layer_inputs[-1] = self.bn(layer_inputs[-1])
        return layer_inputs[-1]
    
# from GNN-DSE https://github.com/UCLA-VAST/GNN-DSE
def create_act(act, num_parameters=None):
    if act == 'relu' or act == 'ReLU':
        return nn.ReLU()
    elif act == 'prelu':
        return nn.PReLU(num_parameters)
    elif act == 'sigmoid':
        return nn.Sigmoid()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'identity' or act == 'None':
        class Identity(nn.Module):
            def forward(self, x):
                return x

        return Identity()
    if act == 'elu' or act == 'elu+1':
        return nn.ELU()
    else:
        raise ValueError('Unknown activation function {}'.format(act))
    
