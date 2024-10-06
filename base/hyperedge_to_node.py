import pandas as pd
import xgi


class HyperedgeToNode():
    def __init__(self, layer_name, directed, file_path, sep="\t") -> None:
        self.layer_name = layer_name
        self.file_path = file_path
        self.directed = directed
        self.sep = sep
        self.df = self._df()
        self.nodes = self._nodes()
        self.hyperedges = self._hyperedges()
        self.hypergraph = self._hypergraph()

    def _df(self):
        df = pd.read_csv(self.file_path, sep=self.sep, index_col=False, dtype=str)
        return df
    
    def _nodes(self):
        assert (not (self.df is None))

        nodes = list()
        for node in list(zip(self.df["node_1"], self.df["node_1_type"], self.df["node_1_name"])):
            nodes.append((node[0], {"node_type": f"{node[1]}", "node_name": node[2]}))
        return nodes
    
    def _hyperedges(self):
        assert (not (self.df is None))

        _hyperedge_1 = self.df["hyperedge_1"].unique()
        _hyperedge_1_type = self.df["hyperedge_1_type"][:len(_hyperedge_1)]
        _hyperedge_1_name = self.df["hyperedge_1_name"].unique()
        
        bipartite_edgelist = list(zip(self.df["node_1"], self.df["hyperedge_1"]))
        _hypergraph = xgi.from_bipartite_edgelist(bipartite_edgelist)
        _hyperedge_list = xgi.to_hyperedge_list(_hypergraph)

        hyperedges = list()
        for hyperedge in list(zip(_hyperedge_list, _hyperedge_1, _hyperedge_1_type, _hyperedge_1_name)):
            hyperedges.append((hyperedge[0], f"{hyperedge[1]}", {"hyperedge_type": f"{hyperedge[2]}", "hyperedge_name": hyperedge[3]}))
        return hyperedges

    def _hypergraph(self):

        if(self.directed):
            raise ValueError("The framework of merging directed hypergraphs into undirected multilayer hypergraphs is not supported for the time being")
        else:
            hypergraph = xgi.Hypergraph()
            hypergraph["name"] = f"{self.layer_name}"
            hypergraph.add_nodes_from(self.nodes)
            hypergraph.add_edges_from(self.hyperedges)
        return hypergraph
    
    @property
    def get_df(self):
        return self.df
    
    @property
    def get_nodes(self):
        return self.nodes
    
    @property
    def get_hyperedges(self):
        return self.hyperedges
    
    @property
    def get_hypergraph(self):
        return self.hypergraph

