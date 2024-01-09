import os
import sys
import torch
import torch_geometric
import random
import pickle
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GraphDataset(torch_geometric.data.Dataset):
    """
        This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
        It can be used in turn by the data loaders provided by pytorch geometric.
        这个类编码一组图，以及一个从磁盘加载此类图的方法。可以依次使用pytorch geometric提供的数据加载程序
        """

    def __init__(self, sample_files):
        # root：string，保存数据集的路径。
        # transform：将Data类型的数据作为输入，并返回转换后的图。数据对象将在每次访问之前进行转换。
        # pre_transform：将Data类型的数据作为输入，并返回转换后的图。数据对象将在保存到硬盘之前进行转换。
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files  # bg，action

    def len(self):  # 返回数据集中的样本数量
        return len(self.sample_files)

    def process_sample(self, filepath):  # 某个样本的处理，获取二分图BG和action
        BGFilepath,solFilepath = filepath  # 二分图和action
        # 读取
        with open(BGFilepath, "rb") as f:
            bgData = pickle.load(f)  # 读取二分图
        with open(solFilepath, "rb") as f:
            solData = pickle.load(f)  # 读取action
        BG = bgData  # 二分图
        return BG,solData

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        此方法加载数据收集期间保存在磁盘上的节点二分图观测。
        """
        BG,sols = self.process_sample(self.sample_files[index])
        A, v_map, v_nodes, c_nodes, b_vars, nor_ob = BG

        constraint_features = c_nodes  # [当前约束系数和/系数均值,当前约束中系数均值,约束右端项,sense]包括目标函数的  表示约束的特征
        edge_indices = A._indices()  # 返回字典中所有的key。 #A: indices_spr第一维放约束编号，第二维放入系数非零的索引, values_spr记录系数不为0的个数,ncons约束数目，nvars获得变量总数
        variable_features = v_nodes  # 节点特征 第一维：[变量编号]，第二维：[变量在约束平均系数,在约束中出现的次数,在约束中系数最大值，在约束中系数最小值]
        edge_features = A._values().unsqueeze(1)  # 取消压缩，添加维度[1,2]->[[1],[2]]

        edge_features = torch.ones(edge_features.shape)
        #constraint_features[np.isnan(constraint_features)] = 1


        graph = BipartiteNodeData(
            torch.FloatTensor(constraint_features),
            torch.LongTensor(edge_indices),
            torch.FloatTensor(edge_features),
            torch.FloatTensor(variable_features),
            nor_ob
        )
        # We must tell pytorch geometric how many nodes there are, for indexing purposes 为了索引，要告诉pytorch geometric有多少个节点
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]  # 一共多少个节点
        graph.ntvars = variable_features.shape[0]
        graph.sols =sols
        graph.files = self.sample_files[index][0]
        graph.edge_ind = edge_indices

        return graph


class PolicyNet(torch.nn.Module):
    def __init__(self, emb_size, cons_nfeats, edge_nfeats, var_nfeats, *args, **kwargs):
        # CONSTRAINT EMBEDDING
        super().__init__(*args, **kwargs)
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),  # 归一化
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.Tanh(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Tanh(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.Tanh(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Tanh(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()

        self.conv_v_to_c2 = BipartiteGraphConvolution()
        self.conv_c_to_v2 = BipartiteGraphConvolution()

        self.output_module = torch.nn.Sequential(  # 输出，接了两个线性输出
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Tanh(),
            torch.nn.Linear(emb_size, 1, bias=False),
            torch.nn.Sigmoid()
        )
        self.to(device)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                # print('before',m.weight)
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                # print(m.weight)

    def forward(
            self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]],
                                            dim=0)  # 沿着一个新维度对输入张量序列进行连接。原来indicate是【约束编号】【非零系数编号】
        # First step: linear embedding layers to a common dimension (64) 先embedding到共同维数
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)

        variable_features = self.var_embedding(variable_features)
        # Two half convolutions 两个半卷积
        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        constraint_features = self.conv_v_to_c2(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v2(
            constraint_features, edge_indices, edge_features, variable_features
        )

        # A final MLP on the variable features
        output = self.output_module(variable_features).squeeze(-1)

        return output

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))
        print('Load success!')


class ValueNet(torch.nn.Module):
    def __init__(self, emb_size, cons_nfeats, edge_nfeats, var_nfeats, *args, **kwargs):
        # CONSTRAINT EMBEDDING
        super().__init__(*args, **kwargs)
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),  # 归一化
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.Tanh(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Tanh(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.Tanh(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Tanh(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()

        self.conv_v_to_c2 = BipartiteGraphConvolution()
        self.conv_c_to_v2 = BipartiteGraphConvolution()

        self.output_module = torch.nn.Sequential(  # 输出，接了两个线性输出
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Tanh(),
            torch.nn.Linear(emb_size, 1, bias=False),
            torch.nn.Tanh()
        )
        self.to(device)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                # print('before',m.weight)
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                # print(m.weight)

    def forward(
            self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]],
                                            dim=0)  # 沿着一个新维度对输入张量序列进行连接。原来indicate是【约束编号】【非零系数编号】

        # First step: linear embedding layers to a common dimension (64) 先embedding到共同维数
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)
        # Two half convolutions 两个半卷积
        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        constraint_features = self.conv_v_to_c2(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v2(
            constraint_features, edge_indices, edge_features, variable_features
        )

        # A final MLP on the variable features

        output = self.output_module(variable_features).squeeze(-1).mean()

        return output

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))
        print('Load success!')


class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    此类以pytorch几何数据处理程序可以理解的格式对“ecole.obstration.NodeBipartite”观察函数返回的节点二分图观察进行编码。
    """

    def __init__(
            self,
            constraint_features,
            edge_indices,
            edge_features,
            variable_features,
            nor_ob
    ):
        super().__init__()  # super()调用父类方法
        self.constraint_features = constraint_features
        self.edge_index = edge_indices  # 实际可以赋值没有报错
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.nor_ob = nor_ob

    def __inc__(self, key, value, store, *args, **kwargs):  # https://blog.51cto.com/u_15717393/5619901
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        我们重载了pytorch几何方法，该方法告诉在连接那些不明显的条目（边索引，候选）的图时如何增加索引。
        """
        if key == "edge_index":
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]  # x.size(0)返回shape的第0维度
            )
        elif key == "candidates":
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    pytorch geometric已经提供了二分图卷积，我们只需要提供所传递消息的确切形式。
    """

    def __init__(self):
        super().__init__("add")
        emb_size = 64

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

        self.post_conv_module = torch.nn.Sequential(torch.nn.LayerNorm(emb_size))

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """
        # https://zhuanlan.zhihu.com/p/427083823?utm_id=0
        # MessagePassing.propagate(edge_index, size=None, **kwargs)：开始传递消息的起始调用，edge_index（边的端点的索引），基于非对称的邻接矩阵进行消息传递（当图为二部图时），需要传递参数size=(N, M)。

        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        # torch.cat（）用于对张量的拼接
        '''
        b = torch.cat([self.post_conv_module(output), right_features], dim=-1)
        a = self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )
        '''

        return self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

    def message(self, node_features_i, node_features_j, edge_features):
        # node_features_i,the node to be aggregated
        # node_features_j,the neighbors of the node i

        # print("node_features_i:",node_features_i.shape)
        # print("node_features_j",node_features_j.shape)
        # print("edge_features:",edge_features.shape)
        #print(len(self.feature_module_left(node_features_i)))
        #print(len(self.feature_module_edge(edge_features)))

        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )

        return output