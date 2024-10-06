import xgi
from .hyperedge_to_node import HyperedgeToNode


class ChemicalToChemical(HyperedgeToNode):
    def __init__(self, layer_name, directed, file_path, sep="\t") -> None:
        super().__init__(layer_name, directed, file_path, sep)


    def _nodes(self):
        assert (not (self.df is None))

        nodes = list()
        for node in list(zip(self.df["node_1"], self.df["node_2"],
                             self.df["node_1_type"], self.df["node_2_name"],
                             self.df["node_1_name"], self.df["node_2_name"])):
            nodes.append((node[0], {"node_type": "chemical", "node_name": node[4]}))
            nodes.append((node[1], {"node_type": "chemical", "node_name": node[5]}))
        return nodes
    
    def _hyperedges(self):
        assert (not (self.df is None))

        bipartite_edgelist = list(zip(self.df["node_1"], self.df["node_2"]))
        
        _hyperedge_id = list(range(len(bipartite_edgelist)))
        hyperedge_id = ["cci: " + str(item) for item in _hyperedge_id]
        hyperedges = list()
        for hyperedge in list(zip(bipartite_edgelist, hyperedge_id, hyperedge_id)):
            hyperedges.append((hyperedge[0], f"{hyperedge[1]}", {"hyperedge_type": "cci"}))
        return hyperedges
    
    def _hypergraph(self):
        if(self.directed):
            hypergraph = xgi.DiHypergraph()
        else: 
            hypergraph = xgi.Hypergraph()
        
        hypergraph["name"] = f"{self.layer_name}"
        hypergraph.add_nodes_from(self.nodes)
        hypergraph.add_edges_from(self.hyperedges)
        return hypergraph