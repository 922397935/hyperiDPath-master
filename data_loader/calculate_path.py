import time
import networkx as nx
import pickle as pkl
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys

sys.path.append('..')


class GraphDataset(object):
    def __init__(self, data_dir, undirected=False):
        self.data_dir = Path(data_dir)
        self.undirected = undirected
        self.data_dict = self._data_loading()
        self.graph = self._network_building()

    def get_network(self):
        return self.graph

    def get_disease_drug_associations(self):
        return self.data_dict['drug_disease']

    def _data_loading(self):
        '''Network'''
        
        grn_pd = pd.read_csv(self.data_dir.joinpath('processed', 'network', 'regulator_gene.csv'))
        
        ppi_pd = pd.read_csv(self.data_dir.joinpath('processed', 'network', 'protein_protein.csv'))
     
        pci_pd = pd.read_csv(self.data_dir.joinpath('processed', 'network', 'chemical_protein.csv'))
    
        cci_pd = pd.read_csv(self.data_dir.joinpath('processed', 'network', 'chemical_chemical.csv'))

        '''Drug and Disease Info'''
        # the columns are target_entrez and disease_icd10
        disease_target_pd = pd.read_csv(self.data_dir.joinpath('processed', 'network', 'indication_protein.csv'))
        # the columns are drug_pubchemcid and target_entrez
        drug_target_pd = pd.read_csv(self.data_dir.joinpath('processed', 'network', 'drug_protein.csv'))
        # the columns are drug_pubchemcid and disease_icd10
        drug_disease_pd = pd.read_csv(self.data_dir.joinpath('processed', 'network','indication_drug.csv'))

        '''data dict'''
        data_dict = {'grn': grn_pd, 'ppi': ppi_pd, 'pci': pci_pd, 'cci': cci_pd, 
                     'disease_target': disease_target_pd, 'drug_target': drug_target_pd,
                     'drug_disease': drug_disease_pd}
        return data_dict
    
    def _df_column_switch(self, df_name):
        # 构造有向边的过程,实现双向
        df_copy = self.data_dict[df_name].copy()
        df_copy.columns = ['from', 'target'] 
        df_switch = self.data_dict[df_name].copy()
        df_switch.columns = ['target', 'from']
        df_concat = pd.concat([df_copy, df_switch])
        df_concat.drop_duplicates(subset=['from', 'target'], inplace=True)
        return df_concat

    def _network_building(self):
        ppi_directed = self._df_column_switch(df_name='ppi')
        pci_directed = self._df_column_switch(df_name='pci')
        cci_directed = self._df_column_switch(df_name='cci')
        
        # the direction in grn is from-gene -> target-gene
        grn_directed = self.data_dict['grn'].copy()
        grn_directed.columns = ['from', 'target']

        if self.undirected:
            print('Creat undirected graph ...')
            drug_target = self._df_column_switch(df_name='drug_target')
            disease_target = self._df_column_switch(df_name='disease_target')
        else:
            print('Creat directed graph ...')
            # the direction in drug-target network is drug -> target
            drug_target = self.data_dict['drug_target'].copy()
            drug_target.columns = ['from', 'target']
            # here the direction in disease-target network should be disease -> target
            disease_target = self.data_dict['disease_target'].copy()
            disease_target.columns = ['from', 'target']
        
        graph_directed = pd.concat([ppi_directed, pci_directed, cci_directed, grn_directed, 
                                    drug_target, disease_target])
        graph_directed.drop_duplicates(subset=['from', 'target'], inplace=True)

        graph_nx = nx.from_pandas_edgelist(graph_directed, source='from', target='target',
                                           create_using=nx.DiGraph())
        
        with open(self.data_dir.joinpath('processed', 'graph', 'graph.pkl'), 'wb') as f:
            pkl.dump(graph_nx, f)

        return graph_nx
    

class GetPath:
    r"""构建多层普通图模型，计算生物路径
    """

    def __init__(self, data_dir):

        self.data_dir = Path(data_dir)
    
        self.candidate_df = pd.read_csv(self.data_dir.joinpath("processed", 'network', "indication_drug.csv"))
        # 在服务器上运行。by_server,drug_path_dict 29G, 401G, indication_path_dict 19G，617G, 本地运行by_local
        self.indication_list, self.drug_list = self._get_indication_drug_list()

    def _get_indication_drug_list(self):
        """返回的是候选集前100的候选集或着全部的候选集, 在计算路径时候需要set一下"""
    
        if self.data_dir.joinpath('processed', 'list', 'drug_list.pkl').is_file() and self.data_dir.joinpath('processed',
                                                                                                        'list', 'indication_list.pkl').is_file():
            print("load existing the whole candidate drug and indication list.pkl")
            drug_list = pkl.load(open(self.data_dir.joinpath('processed', 'list', 'drug_list.pkl'), 'rb'))
            indication_list = pkl.load(open(self.data_dir.joinpath('processed', 'list', 'indication_list.pkl'), 'rb'))
        else:
            print("Create the whole candidate drug and indication list.pkl")
            drug_list = self.candidate_df["node_1"].tolist()
            indication_list = self.candidate_df['hyperedge_1'].tolist()

            indication_list = list(set(indication_list))
            drug_list = list(set(drug_list))
            print("候选疾病集：", len(indication_list))
            print("候选药物集：", len(drug_list))
            with open(f"../data/processed/list/indication_list.pkl", 'wb') as f:
                pkl.dump(indication_list, f)
            with open(f"../data/processed/list/drug_list.pkl", "wb") as f:
                pkl.dump(drug_list, f)

        return indication_list, drug_list

    def get_indication_and_drug_path_dict(self, whether_calculate_path=True):
        r"""在0.1版本中证明s-walk和超图转二部图, 普通图求得的路径都不太行，有弊端
        s-walk计算复杂度高, 运行太久
        超图转二部图, 掺入了虚拟超边，内存需求高，而且很可能对实验结果造成噪声影响
        普通图计算路径能规避虚拟超边的影响，也能囊括各种生物类型
        """
        
        if self.data_dir.joinpath('processed', 'graph', 'graph.pkl').is_file():
            print(f'Load existing graph...')
            self.graph = pkl.load(
                open(self.data_dir.joinpath('processed', 'graph', 'graph.pkl'), 'rb'))
            print(self.graph)
        else:
    
            data = GraphDataset(data_dir=self.data_dir)
            self.graph = data.get_network()
            print(self.graph)

        if whether_calculate_path:

            print("候选疾病集：", len(self.indication_list))
            print("候选药物集：", len(self.drug_list))
            print(f"get indication_path_dict...")

            indication_path_dict = dict()
            for ind in tqdm(self.indication_list):
                ind_dict = nx.single_source_shortest_path(self.graph, ind)  # cut_off限定长度
                del ind_dict[ind]
                indication_path_dict[ind] = ind_dict

            with open(f'../data/path/indication_path_dict.pkl', 'wb') as f:
                pkl.dump(indication_path_dict, f)

            print('get drug_path_dict...')
            
            drug_path_dict = dict()
            for drug in tqdm(self.drug_list):
                drug_dict = nx.single_source_shortest_path(self.graph, drug)
                del drug_dict[drug]
                drug_path_dict[drug] = drug_dict

            with open(f'../data/path/drug_path_dict.pkl', 'wb') as f:
                pkl.dump(drug_path_dict, f)
    

if __name__ == '__main__':
    start = time.time()
    
    def timeit(name):
        print("wall time ({}): {:.0f}s".format(name, time.time() - start))

    path_obj = GetPath(data_dir='../data')  # 为什么得是绝对路径呢？
    path_obj.get_indication_and_drug_path_dict()
    timeit('done')


