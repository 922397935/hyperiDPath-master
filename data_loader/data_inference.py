import os
import pickle
import torch
import random
import numpy as np
import pandas as pd
import xgi
import warnings

from tqdm import tqdm
from scipy import sparse
from pathlib import Path
from base import BaseDataLoader
from data_loader.dataset import Dataset

warnings.filterwarnings("ignore")


class PathDataLoader(BaseDataLoader):
    r"""
    """

    def __init__(
            self,
            data_dir,
            drug_indication_pd_dir,
            max_path_length=8,
            max_path_num=8,
            random_state=0):

        random.seed(0)
        self.data_dir = Path(data_dir)
        self.drug_indication_pd_dir = drug_indication_pd_dir
        self.max_path_length = max_path_length
        self.max_path_num = max_path_num
        self.seed = random_state
    
        self.rng = np.random.RandomState(self.seed)

        self.multilayer_hypergraph = self._data_loader()
        self.node_num = self.multilayer_hypergraph.num_nodes
        self.hyperedge_num = self.multilayer_hypergraph.num_edges
        self.total_type_dict = self._get_type_dict()

        self._load_path_dict()
        self.drug_2index_dict, self.indication_2index_dict = self._get_drug_indication_map_dict()

    def get_node_num(self):
        return self.node_num

    def get_type_num(self):
        return 9
    
    def get_hyperedge_num(self):
        return self.hyperedge_num

    def get_node_map_dict(self):
        r"""这里需要重写吗？感觉不用, 因为映射原理用的是字典, 虽然用的是node_map, 但里面有超边"""
        node_map_pd = pd.read_csv(self.data_dir.joinpath('processed', 'graph', 'node.csv'))
        # TODO: 这里应该可以换成节点的实际名称，出现了偏差是否是因为虚拟节点的存在？
        node_map_dict = {row['map']: row['node'] for _, row in node_map_pd.iterrows()}
        node_map_dict[0] = 0
        return node_map_dict

    def get_sparse_lap_raw_hg(self):
       
        def lap_normalize(mx):
            """Row-normalize sparse matrix"""
            rowsum = np.array(mx.sum(1))
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.
            r_mat_inv = sparse.diags(r_inv)
            mx = r_mat_inv.dot(mx)
            return mx

        def sparse_mx_to_torch_sparse_tensor(sparse_mx):
            """Convert a scipy sparse matrix to a torch sparse tensor."""
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
            indices = torch.from_numpy(
                np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
            values = torch.from_numpy(sparse_mx.data)
            shape = torch.Size(sparse_mx.shape)
            return torch.sparse.FloatTensor(indices, values, shape)

        print('Get raw hyeprgraph sparse lapacian matrix in csr format.')
        # csr matrix, note that if there is a link from node A to B, then the nonzero value in the adjacency matrix is (A, B)
        # where A is the row number and B is the column number

        csr_lapmatrix = xgi.normalized_hypergraph_laplacian(self.multilayer_hypergraph, sparse=True)

        virtual_col = sparse.csr_matrix(np.zeros([self.node_num, 1]))  # 444549个0
        csr_lapmatrix = sparse.hstack([virtual_col, csr_lapmatrix])
        virtual_row = sparse.csr_matrix(np.zeros([1, self.node_num + 1]))
        csr_lapmatrix = sparse.vstack([virtual_row, csr_lapmatrix])

        lap = csr_lapmatrix.tocoo()
        # inc = inc_normalize(inc + sparse.eye(row_num) # 这句话就不适合加了，在超图的卷积公式中关联矩阵不需要加
        lap = lap_normalize(lap)

        lap_tensor = sparse_mx_to_torch_sparse_tensor(lap)
        return lap_tensor
    
    def get_sparse_lap_dual_hg(self):
        r"""
        加了虚拟超边, 构造了对偶超图的拉普拉斯稀疏张量
        """
        def lap_normalize(mx):
            """Row-normalize sparse matrix"""
            rowsum = np.array(mx.sum(1))
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.
            r_mat_inv = sparse.diags(r_inv)
            mx = r_mat_inv.dot(mx)
            return mx

        def sparse_mx_to_torch_sparse_tensor(sparse_mx):
            """Convert a scipy sparse matrix to a torch sparse tensor."""
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
            indices = torch.from_numpy(
                np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
            values = torch.from_numpy(sparse_mx.data)
            shape = torch.Size(sparse_mx.shape)
            return torch.sparse.FloatTensor(indices, values, shape)

        print('Get dual hypergraph sparse lapacian matrix in csr format.')
        # csr matrix, note that if there is a link from node A to B, then the nonzero value in the adjacency matrix is (A, B)
        # where A is the row number and B is the column number

        csr_lapmatrix = xgi.normalized_hypergraph_laplacian(self.multilayer_hypergraph.dual(), sparse=True)

        # TODO: idpath add virtual node (index is 0) 添加虚拟节点
      
        virtual_col = sparse.csr_matrix(np.zeros([self.hyperedge_num, 1]))  # 444549个0
        csr_lapmatrix = sparse.hstack([virtual_col, csr_lapmatrix])
        virtual_row = sparse.csr_matrix(np.zeros([1, self.hyperedge_num + 1]))
        csr_lapmatrix = sparse.vstack([virtual_row, csr_lapmatrix])

        lap = csr_lapmatrix.tocoo()
        # inc = inc_normalize(inc + sparse.eye(row_num) # 这句话就不适合加了，在超图的卷积公式中关联矩阵不需要加
        lap = lap_normalize(lap)

        lap_tensor = sparse_mx_to_torch_sparse_tensor(lap)
        return lap_tensor

    def _data_loader(self):
        # 读取生物多层网络和其他数据
        print('Load biomolecular multilayer hypergraph...')
        if self.data_dir.joinpath('processed',
                                  'hypergraph',
                                  'RGH-DTH-PPH-CPH-CCH-DPH.json').is_file():
            print('Load existing multilayer_hypergraph.json ...')
            multilayer_hypergraph = xgi.read_json(
                self.data_dir.joinpath(
                    'processed',
                    'hypergraph',
                    'RGH-DTH-PPH-CPH-CCH-DPH.json'),
            nodetype=int, edgetype=int)

        else:
            print('Create multilayer_hypergraph.json ...')
            hypergraph_dataset = Dataset(
                data_dir=os.path.join(
                    self.data_dir,
                    'network_raw'))  # 是无向的
            multilayer_hypergraph = hypergraph_dataset.get_cleanup_ite_from_itn_hg
            self.data_dir.joinpath('processed', 'hypergraph').mkdir(exist_ok=False)
            xgi.write_json(multilayer_hypergraph, path=str(self.data_dir.joinpath('processed', 'hypergraph', 'RGH-DTH-PPH-CPH-CCH-DPH.json')))

        return multilayer_hypergraph

    def _get_type_dict(self):
        r"""
        将节点与超边所属的生物学类别转换成数值型，并存储到字典中
        """
        type_mapping_dict = {'tf': 1, 'miRNA': 2, 'gene': 3, 'drug': 4, 'protein': 5, 'chemical': 6, 'indication': 7, 'ppi': 8, 'cci': 9}

        print('Load existing total_type.csv ...')
        total_type_pd = pd.read_csv(
            self.data_dir.joinpath(
                'processed', 'total_type.csv'))
        total_type_dict = {row['total']: type_mapping_dict[row['type']]
                           for idx, row in total_type_pd.iterrows()}

        return total_type_dict

    def _load_path_dict(self):
        r"""
        """
       
        print('Load drug path dict.pkl ...')
        with self.data_dir.joinpath('path', 'drug_path_dict.pkl').open('rb') as f:
            while True:
                try: 
                    drug_path_dict = pickle.load(f)
                except EOFError:
                    break
        print('Load indication path dict.pkl ...')
        with self.data_dir.joinpath('path', 'indication_path_dict.pkl').open('rb') as f:
            while True:
                try: 
                    indication_path_dict = pickle.load(f)
                except EOFError:
                    break
        print('Load test path dict.pkl...')
        with self.data_dir.joinpath('test', 'test_path_dict.pkl').open('rb') as f:
            path_dict = pickle.load(f)

        drug_protein_pd = pd.read_csv(
            self.data_dir.joinpath(
                'processed', 'network', 'drug_protein.csv'))

        indication_protein_pd = pd.read_csv(
            self.data_dir.joinpath(
                'processed', 'network', 'indication_protein.csv'))
  

        self.drug_path_dict = drug_path_dict  # 100个
        self.indication_path_dict = indication_path_dict  # 100个
        self.path_dict = path_dict  # 更少

        self.drug_protein_pd = drug_protein_pd  # 网络中的药物靶标联系
        self.indication_protein_pd = indication_protein_pd  # 网络中的疾病靶标联系

    def _get_drug_indication_map_dict(self):

        drug_map_pd = pd.read_csv(self.data_dir.joinpath('processed', 'mapping', 'drug_map.csv'))
        indication_map_pd = pd.read_csv(self.data_dir.joinpath('processed', 'mapping', 'indication_map.csv'))

        """将药物在网络中的实际编号与药物名联系上
        第一步： 如果药物的实际编号在候选药物编号中存在，则得到它的映射drug_name。疾病也是同样操作
        这个会不会是字典的更新操作？
        """
        drug_2index_dict = {row['drug']: row['map'] for _, row in drug_map_pd.iterrows()
                               if row['map'] in list(self.drug_path_dict.keys())}
        indication_2index_dict = {row['indication']: row['map'] for _, row in indication_map_pd.iterrows()
                                  if row['map'] in list(self.indication_path_dict.keys())}
        # TODO： 这两个有什么差别，感觉是得到所有药物的映射关系
        drug_2index_dict = {row['drug']: row['map'] for _, row in drug_map_pd.iterrows()}
        indication_2index_dict = {row['indication']: row['map'] for _, row in indication_map_pd.iterrows()}

        return drug_2index_dict, indication_2index_dict

    def create_data(self):
        # 这里需要分批吗？
        drug_indication_pd = pd.read_csv(self.data_dir.joinpath('test', self.drug_indication_pd_dir))

        print('Start processing your input data...')
        if len(drug_indication_pd) == 1:
            # 如果测试的药物疾病对不在候选集中
            for _, row in drug_indication_pd.iterrows():
                drug = row['node_1']
                indication = row['hyperedge_1']
                if drug not in self.drug_2index_dict:
                    print(f'Input {drug} not in our dataset!')
                if indication not in self.indication_2index_dict:
                    print(f"Input {indication} not in our dataset!")
                return

        total_path_array, total_type_array = [], []
        total_lengths_array, total_mask_array = [], []
        drug_used, indication_used = [], []
        drug_index_list, indication_index_list = [], []
        for idx, row in tqdm(drug_indication_pd.iterrows()):
            drug, indication = row['node_1'], row['hyperedge_1']  # drug_name indication_name

            if drug not in self.drug_2index_dict:  # 这里说明drug_name在drug_2index_dict
                print(f'Input {drug} not in our dataset!')
                if indication not in self.indication_2index_dict:
                    print(f"Input {indication} not in our dataset!")
                continue
            if indication not in self.indication_2index_dict:
                print(f'Input {indication} not in our dataset!')
                if drug not in self.drug_2index_dict:
                    print(f"Input {drug} not in our dataset!")
                continue

            drug_index = self.drug_2index_dict[drug]  # 这里取的是药物在网络中的index
            indication_index = self.indication_2index_dict[indication]
            drug_index_list.append(drug_index)  # 存到列表里
            indication_index_list.append(indication_index)

            if tuple([drug_index, indication_index]) in self.path_dict:
                # 判断test.csv中药物疾病对在网络中有没有最短路径，若有，则把这个药物疾病索引号化做元组，取路径字典中的值
                # 若没有，则进入else
                path_array = self.path_dict[tuple([drug_index, indication_index])]
                path_list = []
                for p in path_array.tolist():
                    path_list.append([n for n in p if n != 0])  # 把网络嵌入的pad=0删掉
            else:
                # 若没有直接最短路径连接，则在网络中操作，先取药物靶标列表中，该药物对应哪些靶标
                # 再取疾病靶标列表中，该疾病对应哪些靶标  这里如果是利用超图的话，也能实现类似的功能
                drug_protein_list = list(
                    set(self.drug_protein_pd[self.drug_protein_pd['hyperedge_1'] == drug_index]['node_1']))
                indication_protein_list = list(set(
                    self.indication_protein_pd[self.indication_protein_pd['hyperedge_1'] == indication_index]['node_1']))


                if len(self.drug_path_dict[drug_index]) == 0 or len(self.indication_path_dict[indication_index]) == 0:
                    # 在药物列表中，test.csv的药物若是在drug_path_dict没有路径信息，就输出没有找到路径
                    print(f'Cannot find path for {drug}-{indication}')
                    continue

                # 继续下面的药物查找
                # drug_path_list是疾病蛋白列表中满足条件的蛋白质(0),如果这个蛋白质在该药物path_dict中。那就把这个蛋白质
                # 跟那个疾病拼接, 疾病path_path_list也是同样操作
                drug_path_list = [self.drug_path_dict[drug_index][t] + [indication_index] \
                                  for t in indication_protein_list if t in self.drug_path_dict[drug_index]]
                indication_path_list = [self.indication_path_dict[indication_index][t] + [drug_index] \
                                     for t in drug_protein_list if t in self.indication_path_dict[indication_index]]
                indication_path_list = [path[::-1] for path in indication_path_list]
                # all path starts with drug and ends with disease
                path_list = drug_path_list + indication_path_list
                if len(path_list) == 0:
                    print(f'Cannot find enough path for {drug}-{indication}')
                    continue

            '''Sample path'''
            path_array_list, type_array_list, lengths, mask = [], [], [], []
            for path in path_list:
                path = path[: self.max_path_length]
                pad_num = max(0, self.max_path_length - len(path))
                path_array_list.append(path + [0] * pad_num)
                type_array_list.append([self.total_type_dict[n] for n in path] + [0] * pad_num)
                lengths.append(len(path))
                mask.append([1] * len(path) + [0] * pad_num)
            if tuple([drug_index, indication_index]) not in self.path_dict:
                replace = len(path_array_list) < self.max_path_num
                select_idx_list = [idx for idx in
                                   self.rng.choice(len(path_array_list), size=self.max_path_num, replace=replace)]
            else:
                select_idx_list = list(range(self.max_path_num))
            path_array = np.array(
                [[path_array_list[idx] for idx in select_idx_list]])  # shape: [1, path_num, path_length]
            type_array = np.array(
                [[type_array_list[idx] for idx in select_idx_list]])  # shape: [1, path_num, path_length]
            lengths_array = np.array([lengths[idx] for idx in select_idx_list])
            mask_array = np.array([mask[idx] for idx in select_idx_list])

            total_path_array.append(path_array)
            total_type_array.append(type_array)
            total_lengths_array.append(lengths_array)
            total_mask_array.append(mask_array)

            drug_used.append(drug)
            indication_used.append(indication)

        path_feature = torch.from_numpy(np.concatenate(total_path_array, axis=0)).type(torch.LongTensor)
        type_feature = torch.from_numpy(np.concatenate(total_type_array, axis=0)).type(torch.LongTensor)
        lengths = torch.from_numpy(np.concatenate(total_lengths_array)).type(torch.LongTensor)
        mask = torch.from_numpy(np.concatenate(total_mask_array)).type(torch.ByteTensor)

        return drug_index_list, indication_index_list, path_feature, type_feature, lengths, mask, drug_used, indication_used




