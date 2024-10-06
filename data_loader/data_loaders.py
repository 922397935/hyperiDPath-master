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
from data_loader.path_dataset import PathDataset

warnings.filterwarnings("ignore")


class PathDataLoader(BaseDataLoader):
    r"""
    在生物分子多层超图的基础上, 使用超图深度学习方面的卷积模型进行卷积。路径信息用普通图计算结果是最好的
    """
    def __init__(
            self,
            data_dir,
            batch_size,
            max_path_length=10,
            max_path_num=256,
            random_state=0,
            recreate=False,
            use_indication_seed=False,
            shuffle=True,
            validation_split=0.1,
            test_split=0.2,
            num_workers=1,
            training=True):

        random.seed(0)
        self.data_dir = Path(data_dir)
        self.max_path_length = max_path_length
        self.max_path_num = max_path_num
        self.random_state = random_state
        self.recreate = recreate
        self.use_indication_seed = use_indication_seed
       
        self.rng = np.random.RandomState(random_state)

        self.multilayer_hypergraph = self._data_loader()
        print("multilayer hypergraph's info: ", self.multilayer_hypergraph)
        self.node_num = self.multilayer_hypergraph.num_nodes
        self.hyperedge_num = self.multilayer_hypergraph.num_edges

        self.total_type_dict = self._get_type_dict()

        self.path_dict = self._load_path_dict()

        self.dataset = self._create_dataset()

        super().__init__(
            self.dataset,
            batch_size,
            shuffle,
            validation_split,
            test_split,
            num_workers)

    def get_node_num(self):
        return self.node_num

    def get_hyperedge_num(self):
        return self.hyperedge_num

    def get_type_num(self):
        return 9

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

        # 这种读取的超图，id是字符类型的，怎么避免？
        # 如果没有预先生成的文件，则调用类的实例与函数来生成多层生物网络模型
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
        将节点与超边所属的生物学类别转换成数值型，并存储到字典中,这里返回的type需不需要分开?要不要分的这么细致？
        """
        type_mapping_dict = {'tf': 1, 'miRNA': 2, 'gene': 3, 'drug': 4, 'protein': 5, 'chemical': 6, 'indication': 7, 'ppi': 8, 'cci': 9}
        if self.data_dir.joinpath('processed', 'total_type.csv').is_file():
            print('Load existing total_type.csv ...')
            total_type_pd = pd.read_csv(
                self.data_dir.joinpath(
                    'processed', 'total_type.csv'))
            total_type_dict = {row['total']: type_mapping_dict[row['type']]
                         for idx, row in total_type_pd.iterrows()}
        else:
            print('Create total_type.csv ...')
            hg = self.multilayer_hypergraph.copy()
            # 可以用multi().aspandas()也可以用filterby_attr()来筛选filterby() attrs()
            _node_type_dict = hg.nodes.attrs('node_type').asdict()
            _hyperedge_type_dict = hg.edges.attrs('hyperedge_type').asdict()
            _total_type_dict = _node_type_dict.copy()
            _total_type_dict.update(_hyperedge_type_dict)

            _total_type_pd = pd.DataFrame({'total': list(_total_type_dict.keys()),
                                          'type': list(_total_type_dict.values())})

            total_type_dict = {key: type_mapping_dict[value]
                                   for key, value in _total_type_dict.items()
                                   if value in type_mapping_dict}

            _total_type_pd.to_csv(self.data_dir.joinpath('processed', 'total_type.csv'), index=False)

        return total_type_dict

    def _negative_sampling(self, positive_drug_indication_pd):
        """
        对正样本的药物适应症对进行负采样，得到负采样后的药物适应症
        Args:
            positive_drug_disease_pd:

        Returns:

        """
        print('Negative sampling...')
        drug_list = list(set(positive_drug_indication_pd['node_1']))
        indication_list = list(set(positive_drug_indication_pd['hyperedge_1']))
        negative_drug_list, negative_indication_list = [], []

        if self.use_indication_seed:
            # 用疾病作为负采样的种子
            print('Use indication as seed.')
            for indication in indication_list:
                positive_drug_list = list(
                    positive_drug_indication_pd[positive_drug_indication_pd['hyperedge_1'] == indication]['node_1'])
                drug_left_list = list(set(drug_list) - set(positive_drug_list))
                # random select the drugs with the same number of that of
                # positive drugs
                negative_drug_list += random.sample(drug_left_list, min(
                    len(positive_drug_list), len(drug_left_list)))
                negative_indication_list += [indication] * \
                    min(len(positive_drug_list), len(drug_left_list))
        else:
            print('Use drug as seed.')
            for drug in drug_list:
                positive_indication_list = list(
                    positive_drug_indication_pd[positive_drug_indication_pd['node_1'] == drug]['hyperedge_1'])
                indication_left_list = list(
                    set(indication_list) - set(positive_indication_list))
                # random select the diseases with the same number of that of
                # positive diseases
                negative_indication_list += random.sample(indication_left_list, min(
                    len(positive_indication_list), len(indication_left_list)))
                negative_drug_list += [drug] * \
                    min(len(positive_indication_list), len(indication_left_list))

        negative_pd = pd.DataFrame({'node_1': negative_drug_list,
                                    'hyperedge_1': negative_indication_list})
        print('For {0} drugs, {1} indications negative samples are generated.'.format(
            len(drug_list), len(negative_indication_list)))
        return negative_pd

    def _load_path_dict(self):
        r"""
        """
        if not self.recreate and self.data_dir.joinpath(
                'path',  'path_dict.pkl').is_file():
            print('Load existing path_dict.pkl ...')
            with self.data_dir.joinpath('path', 'path_dict.pkl').open('rb') as f:
                path_dict = pickle.load(f)
        else:
            print('Start creating path_dict ...')
            print(f'Load drug_path_dict.pkl ...')
            
            with self.data_dir.joinpath('path', 'drug_path_dict.pkl').open('rb') as f:
                while True:
                    try: 
                        drug_path_dict = pickle.load(f)
                    except EOFError:
                        break
            print(f'Load indication_path_dict.pkl ...')
            
            with self.data_dir.joinpath('path', 'indication_path_dict.pkl').open('rb') as f:
                while True:
                    try: 
                        indication_path_dict = pickle.load(f)
                    except EOFError:
                        break
            
            positive_drug_indication_pd = pd.read_csv(
                self.data_dir.joinpath(
                    'processed', 'network', 'indication_drug.csv'))  

            drug_protein_pd = pd.read_csv(
                self.data_dir.joinpath(
                    'processed', 'network', 'drug_protein.csv'))

            indication_protein_pd = pd.read_csv(
                self.data_dir.joinpath(
                    'processed', 'network', 'indication_protein.csv'))

            # 负采样后的药物疾病对
            negative_drug_indication_pd = self._negative_sampling(
                positive_drug_indication_pd)
            # create path_dict
            path_dict, drug_indication_pd = self._create_path(positive_drug_indication_pd, negative_drug_indication_pd,
                                                        drug_path_dict, indication_path_dict,
                                                        drug_protein_pd, indication_protein_pd)
            # save 保存的是正样本和负样本的药物疾病对
            with self.data_dir.joinpath('path', 'path_dict.pkl').open('wb') as f:
                pickle.dump(path_dict, f)
                
            drug_indication_pd.to_csv(
                self.data_dir.joinpath(
                    'processed',
                    'drug_indication_negatived.csv'),
                index=False)

        return path_dict

    def _create_path(self, positive_drug_indication_pd, negative_drug_indication_pd,
                     drug_path_dict, indication_path_dict,
                     drug_protein_pd, indication_protein_pd):
        """
        """
        print('Create all the shortest paths between drugs and indications...')
        positive_drug_indication_pd['label'] = [1] * len(positive_drug_indication_pd)
        negative_drug_indication_pd['label'] = [0] * len(negative_drug_indication_pd)
        drug_indication_pd = pd.concat(
            [positive_drug_indication_pd, negative_drug_indication_pd])

        path_dict = dict()
        for idx, row in tqdm(drug_indication_pd.iterrows()):
            drug, indication, label = row['node_1'], row['hyperedge_1'], row['label']
            drug_protein_list = list(
                set(drug_protein_pd[drug_protein_pd['hyperedge_1'] == drug]['node_1']))
            indication_protein_list = list(set(
                indication_protein_pd[indication_protein_pd['hyperedge_1'] == indication]['node_1']))

            # 如果我分批处理的话，这里毫无疑问会报错，字典中没有那个键，限定drug_indication_pd的范围
            if len(drug_path_dict[drug]) == 0 or len(indication_path_dict[indication]) == 0:
                continue
            # TODO: 这里我的模型拼接的路径为什么会差别这么大？
            """
            我的路径为什么中间会经过疾病。。。？没少，PPI, CCI超边的存在,令路径长度从4升到5， 从6升到7，所以分布变成了3,5,7,9
            """
            drug_path_list = [drug_path_dict[drug][t] + [indication]
                              for t in indication_protein_list if t in drug_path_dict[drug]]
            indication_path_list = [indication_path_dict[indication][t] + [drug]
                                 for t in drug_protein_list if t in indication_path_dict[indication]]
            indication_path_list = [path[::-1] for path in indication_path_list]
            # all path starts with drug and ends with disease
            # TODO: 用的字符串连接的操作？
            path_list = drug_path_list + indication_path_list
            if len(path_list) == 0:
                continue
            path_dict[tuple([drug, indication, label])] = path_list
            # path_dict是以药物，疾病，label为标志的
        return path_dict, drug_indication_pd

    def _create_dataset(self):
        print('Creating tensor dataset...')
        drug_indication_array = list(self.path_dict.keys())

        dataset = PathDataset(drug_indication_array=drug_indication_array,
                              total_path_dict=self.path_dict,
                              type_dict=self.total_type_dict,
                              max_path_length=self.max_path_length,
                              max_path_num=self.max_path_num,
                              rng=self.rng)
       
     
        if not self.recreate and self.data_dir.joinpath("test", "test_path_dict.pkl").is_file():
            pass
        else:
            # 对正负样本的药物疾病对打乱了顺序
            print('Start creating test path dict ...')
            test_path_dict = dict()
            for i in range(len(dataset)):
                drug, disease, path_feature = dataset[i][0], dataset[i][1], dataset[i][2]
                test_path_dict[tuple([drug, disease])] = np.array(path_feature)
            test_drug_disease_array = list(test_path_dict.keys())
            random.shuffle(test_drug_disease_array)
            random_test_path_dict = dict()
            for key in test_drug_disease_array:
                random_test_path_dict[key] = test_path_dict.get(key)

            with open(self.data_dir.joinpath("test", "test_path_dict.pkl"), "wb") as f:
                pickle.dump(random_test_path_dict, f)

        return dataset

    def create_path_for_repurposing(self, indication, total_test_drug):
        # 对药物遍历，找能与它能相连的疾病
        total_path_array, total_type_array, label = [], [], []
        total_lengths_array, total_mask_array = [], []
        drug_used = []
        for drug in total_test_drug:
            '''Find all the path'''
            drug_protein_list = list(set(
                self.drug_protein_pd[self.drug_protein_pd['hyperedge_1'] == drug]['node_1']))
            indication_protein_list = list(set(
                self.indication_protein_pd[self.indication_protein_pd['hyperedge_1'] == indication]['node_1']))
            if len(self.drug_path_dict[drug]) == 0 or len(
                    self.indication_path_dict[indication]) == 0:
                continue
            drug_path_list = [self.drug_path_dict[drug][t] + [indication]
                              for t in indication_protein_list if t in self.drug_path_dict[drug]]
            indication_path_list = [
                self.indication_path_dict[indication][t] +
                [drug] for t in drug_protein_list if t in self.indication_path_dict[indication]]
            indication_path_list = [path[::-1] for path in indication_path_list]
            # all path starts with drug and ends with disease
            path_list = drug_path_list + indication_path_list

            '''Sample path'''
            path_array_list, type_array_list, lengths, mask = [], [], [], []
            for path in path_list:
                path = path[: self.max_path_length]
                pad_num = max(0, self.max_path_length - len(path))
                path_array_list.append(path + [0] * pad_num)
                type_array_list.append([self.total_type_dict[n]
                                        for n in path] + [0] * pad_num)
                lengths.append(len(path))
                mask.append([1] * len(path) + [0] * pad_num)
            replace = len(path_array_list) < self.max_path_num
            select_idx_list = [
                idx for idx in self.rng.choice(
                    len(path_array_list),
                    size=self.max_path_num,
                    replace=replace)]
            # shape: [1, path_num, path_length] 1 256 8
            path_array = np.array([[path_array_list[idx]
                                    for idx in select_idx_list]])
            # shape: [1, path_num, path_length] 1 256 8
            type_array = np.array([[type_array_list[idx]
                                    for idx in select_idx_list]])
            lengths_array = np.array([lengths[idx] for idx in select_idx_list])
            mask_array = np.array([mask[idx] for idx in select_idx_list])

            total_path_array.append(path_array)
            total_type_array.append(type_array)
            total_lengths_array.append(lengths_array)
            total_mask_array.append(mask_array)
            if drug in self.indication2drug_dict[indication]:
                label.append([1])
            else:
                label.append([0])

            drug_used.append(drug)
       
        path_feature = torch.from_numpy(
            np.concatenate(
                total_path_array,
                axis=0)).type(
            torch.LongTensor)
        type_feature = torch.from_numpy(
            np.concatenate(
                total_type_array,
                axis=0)).type(
            torch.LongTensor)
        label = torch.from_numpy(np.array(label)).type(torch.FloatTensor)
        lengths = torch.from_numpy(
            np.concatenate(total_lengths_array)).type(
            torch.LongTensor)
        mask = torch.from_numpy(
            np.concatenate(total_mask_array)).type(
            torch.ByteTensor)

        return path_feature, type_feature, lengths, mask, label, drug_used

    def get_recommendation_data(self):
        '''Get the unique drug and disease in the test dataset'''
       
        drug_indication_array = list(self.path_dict.keys())
        test_drug_indication_array = [drug_indication_array[idx]
                                    for idx in self.test_idx]
        print(
            '{} records in test dataset'.format(
                len(test_drug_indication_array)))
        total_test_drug, total_test_indicaiton = [], []
        indication2drug_dict = dict()
        for drug, indication, label in test_drug_indication_array:
            if label == 0:
                continue
            total_test_drug.append(drug)
            total_test_indicaiton.append(indication)
            if indication not in indication2drug_dict:
                indication2drug_dict[indication] = [drug]
            else:
                indication2drug_dict[indication].append(drug)
        total_test_drug = list(set(total_test_drug))
        total_test_indicaiton = list(set(total_test_indicaiton))
        self.indication2drug_dict = {
            indication: list(
                set(drug_list)) for indication,
                                    drug_list in indication2drug_dict.items()}
        '''Prepare the dataset'''
        print('Start creating path_dict for test dataset...')
        print('Load drug_path_dict.pkl ...')
     
        with self.data_dir.joinpath('path', 'drug_path_dict.pkl').open('rb') as f:
            while True:
                try: 
                    self.drug_path_dict = pickle.load(f)
                except EOFError:
                    break
        print('Load indication_path_dict.pkl ...')
        with self.data_dir.joinpath('path', 'indication_path_dict.pkl').open('rb') as f:
            while True:
                try: 
                    self.indication_path_dict = pickle.load(f)
                except EOFError:
                    break

        self.drug_protein_pd = pd.read_csv(
            self.data_dir.joinpath(
                'processed',
                'network',
                'drug_protein.csv'))

        self.indication_protein_pd = pd.read_csv(
            self.data_dir.joinpath(
                'processed',
                'network',
                'indication_protein.csv'))
        
        return total_test_drug, total_test_indicaiton
        
        

