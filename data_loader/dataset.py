import xgi
from tqdm import tqdm
import sys

sys.path.append('..')
from copy import deepcopy
from base.regulator_to_gene import RegulatorToGene
from base.drug_to_protein import DrugToProtein
from base.protein_to_protein import ProteinToProtein
from base.indication_to_protein import IndicationToProtein
from base.biological_function_to_protein import BiologicalFunctionToProtein
from base.biological_function_to_biological_function import BiologicalFunctionToBiologicalFunction
from base.indication_to_drug import IndicationToDrug
from base.chemical_to_chemical import ChemicalToChemical
from base.chemical_to_protein import ChemicalToProtein

GENE = 'gene'
REGULATOR = 'regulator'
DRUG = 'drug'
INDICATION = 'indication'
PROTEIN = 'protein'
BIOLOGICAL_FUNCTION = 'biological_function'
CHEMICAL = 'chemical'

REGULATOR_GENE = 'regulator-gene'
CHEMICAL_CHEMICAL = 'chemical-chemical'
CHEMICAL_PROTEIN = 'chemical-protein'
DRUG_PROTEIN = 'drug-protein'
PROTEIN_PROTEIN = 'protein-protein'
INDICATION_PROTEIN = 'indication-protein'
BIOLOGICAL_FUNCTION_PROTEIN = 'biological_function-protein'
BIOLOGICAL_FUNCTION_BIOLOGICAL_FUNCTION = 'biological_function-biological_function'
INDICATION_DRUG = 'indication-drug'


class Dataset:
    r"""
    构建生物分子多层超网络(Undirected):
    1. 基因-调控超网络: RegNetwork(expert) 调控因子为超边, 靶基因为节点
    2. 药物-靶标超网络: msi, 药物为超边, 靶标为节点
    3. 蛋白-蛋白相互作用超网络 (2-均匀超图）: msi(included corum) 超边表示PPI的相互作用
    4. 生物功能-蛋白超网络: 生物功能为超边，蛋白为节点
    5.生物功能-生物功能超网络 (2-均匀超图）: 超边表征生物功能的层次关系，
    6. 化学分子-蛋白质超网络: 化学分子为超边, 蛋白质为节点
    7. 化学分子-化学分子超网络: 化学分子2均匀超图
    8. 疾病-靶基因超网络: INDICATION为超边, 靶gene为节点
    
    9. TODO: 药物疾病数据: 疾病为超边, 药物为节点, 作为候选集
    """

    def __init__(self,
                 data_dir,
                 name: str = 'RGH-DTH-PPH-CPH-CCH-DPH',
                 directed=False,
                 nodes=[GENE, PROTEIN, BIOLOGICAL_FUNCTION, DRUG, CHEMICAL],
                 hyperedges=[REGULATOR_GENE, DRUG_PROTEIN, PROTEIN_PROTEIN, INDICATION_PROTEIN,
                             BIOLOGICAL_FUNCTION_PROTEIN, BIOLOGICAL_FUNCTION_BIOLOGICAL_FUNCTION, INDICATION_DRUG,
                             CHEMICAL_PROTEIN, CHEMICAL_CHEMICAL]):

        self.data_dir = data_dir
        self.name = name
        self.directed = directed
        self.nodes = nodes
        self.hyperedges = hyperedges

        self.regulator2gene_file_path = f'{data_dir}/0_regulator_to_gene.tsv'
        self.drug2protein_file_path = f'{data_dir}/1_drug_to_protein.tsv'
        self.indication2protein_file_path = f'{data_dir}/2_indication_to_protein.tsv'
        self.protein2protein_file_path = f'{data_dir}/3_protein_to_protein.tsv'
        self.biological_function2protein_file_path = f'{data_dir}/4_biological_function_to_protein.tsv'
        self.biological_function2biological_function_file_path = f'{data_dir}/5_biological_function_to_biological_function.tsv'
        self.indication2drug_file_path = f'{data_dir}/6_indication_to_drug.tsv'
        self.chemical2protein_file_path = f'{data_dir}/7_chemical_to_protein.tsv'
        self.chemical2chemical_file_path = f'{data_dir}/8_chemical_to_chemical.tsv'

        # 每层的方向，每层都是无向，如果单层要设置有向的可以个性化定义某层的方向
        self.regulator2gene_directed = self.directed
        self.drug2protein_directed = self.directed
        self.indication2protein_directed = self.directed
        self.protein2protein_directed = self.directed
        self.biological_function2protein_directed = self.directed
        self.biological_function2biological_function_directed = self.directed
        self.indication2drug_directed = self.directed
        self.chemical2protein_directed = self.directed
        self.chemical2chemical_directed = self.directed

        self.each_layer_hypergraph = self._each_layer_hypergraph_building()
        self.multilayer_hypergraph = self._multilayer_hypergraph_building()
        self.cleanup_ite_from_itn_hg = self._get_cleanup_ite_from_itn_hg()
       

    def _multilayer_hypergraph_building(self):
        r"""构造生物分子多层超图"""
        multilayer_hypergraph = xgi.Hypergraph()

        for key in tqdm(self.each_layer_hypergraph.keys()):
            # 这里可以选择想用哪几层组成超网络,
            # 我想要的path、是从药物超边，能遍历节点与其他超边到达疾病超边的路径
            if key not in [INDICATION_DRUG, BIOLOGICAL_FUNCTION_BIOLOGICAL_FUNCTION, BIOLOGICAL_FUNCTION_PROTEIN]: # 去掉候选集与生物功能层的多层超图
            # if key in [DRUG_PROTEIN, PROTEIN_PROTEIN, INDICATION_PROTEIN]:  # 可以选择几层组成多层超图
                multilayer_hypergraph.add_nodes_from(self.each_layer_hypergraph[key].nodes)
                multilayer_hypergraph.add_edges_from(self.each_layer_hypergraph[key].hyperedges)
        multilayer_hypergraph["name"] = self.name

        return multilayer_hypergraph

    def _each_layer_hypergraph_building(self):
        r"""每层生物超图构建，返回超图字典"""
        hypergraph_dict = dict()

        if (GENE in self.nodes) and (REGULATOR_GENE in self.hyperedges):
            hypergraph_dict[REGULATOR_GENE] = RegulatorToGene(REGULATOR_GENE, self.regulator2gene_directed,
                                                              self.regulator2gene_file_path)
        if (PROTEIN in self.nodes) and (DRUG_PROTEIN in self.hyperedges):
            hypergraph_dict[DRUG_PROTEIN] = DrugToProtein(DRUG_PROTEIN, self.drug2protein_directed,
                                                          self.drug2protein_file_path)
        if (PROTEIN in self.nodes) and (PROTEIN_PROTEIN in self.hyperedges):
            hypergraph_dict[PROTEIN_PROTEIN] = ProteinToProtein(PROTEIN_PROTEIN, self.protein2protein_directed,
                                                                self.protein2protein_file_path)     
        if (PROTEIN in self.nodes) and (CHEMICAL_PROTEIN in self.hyperedges):
            hypergraph_dict[CHEMICAL_PROTEIN] = ChemicalToProtein(CHEMICAL_PROTEIN,
                                                                  self.chemical2protein_directed,
                                                                  self.chemical2protein_file_path)
        if (CHEMICAL in self.nodes) and (CHEMICAL_CHEMICAL in self.hyperedges):
            hypergraph_dict[CHEMICAL_CHEMICAL] = ChemicalToChemical(CHEMICAL_CHEMICAL,
                                                                    self.chemical2chemical_directed,
                                                                    self.chemical2chemical_file_path)
        if (PROTEIN in self.nodes) and (BIOLOGICAL_FUNCTION_PROTEIN in self.hyperedges):
            hypergraph_dict[BIOLOGICAL_FUNCTION_PROTEIN] = BiologicalFunctionToProtein(BIOLOGICAL_FUNCTION_PROTEIN,
                                                                                       self.biological_function2protein_directed,
                                                                                       self.biological_function2protein_file_path)
        if (BIOLOGICAL_FUNCTION in self.nodes) and (BIOLOGICAL_FUNCTION_BIOLOGICAL_FUNCTION in self.hyperedges):
            hypergraph_dict[BIOLOGICAL_FUNCTION_BIOLOGICAL_FUNCTION] = BiologicalFunctionToBiologicalFunction(
                BIOLOGICAL_FUNCTION_BIOLOGICAL_FUNCTION,
                self.biological_function2biological_function_directed,
                self.biological_function2biological_function_file_path)
        if (PROTEIN in self.nodes) and (INDICATION_PROTEIN in self.hyperedges):
            hypergraph_dict[INDICATION_PROTEIN] = IndicationToProtein(INDICATION_PROTEIN,
                                                                      self.indication2protein_directed,
                                                                      self.indication2protein_file_path)

        if (DRUG in self.nodes) and (INDICATION_DRUG in self.hyperedges):
            hypergraph_dict[INDICATION_DRUG] = IndicationToDrug(INDICATION_DRUG, self.indication2drug_directed,
                                                                self.indication2drug_file_path)

        return hypergraph_dict

    def display_model_info(self):
        r"""显示模型信息"""
        statics = self.each_layer_hypergraph
        for key in statics.keys():
            print(f"{statics[key].hypergraph}")
        print(self.multilayer_hypergraph)
        with open('../data/processed/hypergraph/info.txt', 'w') as file:
            # 将标准输出重定向到文件
            sys.stdout = file
            # 在控制台打印信息
            for key in statics.keys():
                print(f"{statics[key].hypergraph}")
            print(self.multilayer_hypergraph)
            # 恢复标准输出
            sys.stdout = sys.__stdout__

    def save_hypergraph(self, path="../data/processed"):
        r"""保存重新编号后的超图张量数据"""
        node_df = self.cleanup_ite_from_itn_hg.nodes.multi([self.cleanup_ite_from_itn_hg.nodes.attrs("label"),
                                                            self.cleanup_ite_from_itn_hg.nodes.attrs("node_name"),
                                                            self.cleanup_ite_from_itn_hg.nodes.attrs("node_type")]).aspandas()
        hyperedge_df = self.cleanup_ite_from_itn_hg.edges.multi([self.cleanup_ite_from_itn_hg.edges.attrs("label"),
                                                                 self.cleanup_ite_from_itn_hg.edges.attrs("hyperedge_name"),
                                                                 self.cleanup_ite_from_itn_hg.edges.attrs("hyperedge_type")]).aspandas()

        node_df.to_csv(f"{path}/hypergraph/node.csv")
        hyperedge_df.to_csv(f"{path}/hypergraph/hyperedge.csv")

        xgi.write_json(self.cleanup_ite_from_itn_hg, f"{path}/hypergraph/{self.name}.json")

    @staticmethod
    def convert_labels_to_integers(net, node_index=0, label_attribute="label"):
        """
        节点从node_index开始编号, 默认从0开始
            node_index=0: 节点从0开始编号, 超边跟着到n, type编号从0开始, 这种编号方式方便对接deephypergraph
            node_index=1: 节点从1开始编号, 超边跟着到n, type编号就从1开始
        """

        node_dict = dict(zip(net.nodes, range(node_index, net.num_nodes + node_index)))
        edge_dict = dict(zip(net.edges, range(net.num_nodes + node_index, net.num_nodes + net.num_edges + node_index)))

        temp_net = net.__class__()
        temp_net._hypergraph = deepcopy(net._hypergraph)

        temp_net.add_nodes_from((id, deepcopy(net.nodes[n])) for n, id in node_dict.items())
        temp_net.set_node_attributes(
            {n: {label_attribute: id} for id, n in node_dict.items()}
        )

        temp_net.add_edges_from(
            (
                {node_dict[n] for n in e},
                edge_dict[id],
                deepcopy(net.edges[id]),
            )
            for id, e in net.edges.members(dtype=dict).items()
        )

        temp_net.set_edge_attributes(
            {e: {label_attribute: id} for id, e in edge_dict.items()}
        )

        return temp_net

    def cleanup_ite_from_itn(
            self,
            isolates=False,
            singletons=False,
            multiedges=False,
            connected=True,
            relabel=True,
            in_place=True,
    ):

        if in_place:
            if not multiedges:
                self.multilayer_hypergraph.merge_duplicate_edges()
            if not singletons:
                self.multilayer_hypergraph.remove_edges_from(self.multilayer_hypergraph.edges.singletons())
            if not isolates:
                self.multilayer_hypergraph.remove_nodes_from(self.multilayer_hypergraph.nodes.isolates())
            if connected:
                from xgi.algorithms import largest_connected_component

                self.multilayer_hypergraph.remove_nodes_from(self.multilayer_hypergraph.nodes - largest_connected_component(self.multilayer_hypergraph))
            if relabel:
                temp = Dataset.convert_labels_to_integers(self.multilayer_hypergraph, node_index=1).copy()

                nn = temp.nodes
                ee = temp.edges

                self.multilayer_hypergraph.clear()
                self.multilayer_hypergraph.add_nodes_from((n, deepcopy(attr)) for n, attr in nn.items())
                self.multilayer_hypergraph.add_edges_from(
                    (e, id, deepcopy(temp.edges[id]))
                    for id, e in ee.members(dtype=dict).items()
                )
                self._hypergraph = deepcopy(temp._hypergraph)
        else:
            H = self.multilayer_hypergraph.copy()
            if not multiedges:
                H.merge_duplicate_edges()
            if not singletons:
                H.remove_edges_from(H.edges.singletons())
            if not isolates:
                H.remove_nodes_from(H.nodes.isolates())
            if connected:
                from xgi.algorithms import largest_connected_component

                H.remove_nodes_from(H.nodes - largest_connected_component(H))
            if relabel:
                H = Dataset.convert_labels_to_integers(H, node_index=1)

            return H

    def _get_cleanup_ite_from_itn_hg(self):
        r"""
        返回超边索引从节点数开始之后的超图
        """
        cleanup_ite_from_itn_hg = self.cleanup_ite_from_itn(isolates=True, singletons=True,
                                                            multiedges=True, connected=False,
                                                            relabel=True, in_place=False)
        return cleanup_ite_from_itn_hg

    @property
    def get_each_layer_hypergraph(self):
        return self.each_layer_hypergraph

    @property
    def get_multilayer_hypergraph(self):
        return self.multilayer_hypergraph

    @property
    def get_cleanup_ite_from_itn_hg(self):
        return self.cleanup_ite_from_itn_hg


if __name__ == '__main__':

    hg = Dataset(data_dir='../data/network_raw/')
    hg.display_model_info()
    hg.save_hypergraph()
    
    

