from .protein_to_protein import ProteinToProtein


class BiologicalFunctionToBiologicalFunction(ProteinToProtein):
    def __init__(self, layer_name, directed, file_path, sep="\t") -> None:
        super().__init__(layer_name, directed, file_path, sep)
        
    def _hyperedges(self):
        assert (not (self.df is None))

        bipartite_edgelist = list(zip(self.df["node_1"], self.df["node_2"]))
        
        _hyperedge_id = list(range(len(bipartite_edgelist)))
        hyperedge_id = ["bbi: " + str(item) for item in _hyperedge_id]
        hyperedges = list()
        for hyperedge in list(zip(bipartite_edgelist, hyperedge_id)):
            hyperedges.append((hyperedge[0], f"{hyperedge[1]}", {"hyperedge_type": "bbi"}))
        return hyperedges