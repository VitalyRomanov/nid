from dataclasses import dataclass
from typing import List, Dict, Union


@dataclass
class ParsedGraph:
    nodes: List[Dict[str, Union[int, str]]]
    edges: List[Dict[str, Union[int, str]]]

    def as_df(self):
        import pandas as pd
        nodes = pd.DataFrame.from_records(self.nodes)
        edges = pd.DataFrame.from_records(self.edges)

        return nodes, edges
