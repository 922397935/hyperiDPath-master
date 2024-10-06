import xgi

from .hyperedge_to_node import HyperedgeToNode


class RegulatorToGene(HyperedgeToNode):
    def __init__(self, layer_name, directed, file_path, sep="\t") -> None:
        super().__init__(layer_name, directed, file_path, sep)
    
    def _hyperedges(self):
        assert (not (self.df is None))

        _hyperedge_1 = self.df["hyperedge_1"].unique()
        # 这里超边type得将TF和miRNA分开 1466
        _hyperedge_1_type = ['tf'] * 1407 + ['miRNA'] * (len(_hyperedge_1) - 1407)
        temp = list(self.df["hyperedge_1_name"].unique())
        _hyperedge_1_name = temp
        
        bipartite_edgelist = list(zip(self.df["node_1"], self.df["hyperedge_1"]))
        _hypergraph = xgi.from_bipartite_edgelist(bipartite_edgelist)
        _hyperedge_list = xgi.to_hyperedge_list(_hypergraph)

        hyperedges = list()
        for hyperedge in list(zip(_hyperedge_list, _hyperedge_1, _hyperedge_1_type, _hyperedge_1_name)):
            hyperedges.append((hyperedge[0], f"{hyperedge[1]}", {"hyperedge_type": f"{hyperedge[2]}", "hyperedge_name": hyperedge[3]}))
        return hyperedges
