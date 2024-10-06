import xgi
from .hyperedge_to_node import HyperedgeToNode


class IndicationToDrug(HyperedgeToNode):
    def __init__(self, layer_name, directed, file_path, sep="\t") -> None:
        super().__init__(layer_name, directed, file_path, sep)

    def _hyperedges(self):

        _hyperedge_1 = self.df["hyperedge_1"].unique()
        _hyperedge_1 = ["(NLM UMLS CUIDs: " + str(item) for item in _hyperedge_1]
        _hyperedge_1_type = self.df["hyperedge_1_type"][:len(_hyperedge_1)]
        _hyperedge_1_name = self.df["hyperedge_1_name"].unique()
        
        bipartite_edgelist = list(zip(self.df["node_1"], self.df["hyperedge_1"]))
        _hypergraph = xgi.from_bipartite_edgelist(bipartite_edgelist)
        _hyperedge_list = xgi.to_hyperedge_list(_hypergraph)

        hyperedges = list()
        for hyperedge in list(zip(_hyperedge_list, _hyperedge_1, _hyperedge_1_type, _hyperedge_1_name)):
            hyperedges.append((hyperedge[0], f"{hyperedge[1]}", {"hyperedge_type": f"{hyperedge[2]}", "hyperedge_name": hyperedge[3]}))
        return hyperedges