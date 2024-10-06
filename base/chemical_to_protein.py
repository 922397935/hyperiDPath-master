from .hyperedge_to_node import HyperedgeToNode


class ChemicalToProtein(HyperedgeToNode):
    def __init__(self, layer_name, directed, file_path, sep="\t") -> None:
        super().__init__(layer_name, directed, file_path, sep)