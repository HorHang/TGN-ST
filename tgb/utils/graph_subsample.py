import numpy as np
from typing import Union, Optional

### find which is the appropriate input data structure ###

def subsampling(graph: Union[object, dict], 
                      node_list: Optional[list] = [], 
                      random_selection: Optional[bool] = False, 
                      N: Optional[int] = 100
                      ) -> dict:
    """
    Subsampling a part of graph by only monitoring the contacts from specific nodes' list

    Parameters:
        graph: graph object or edgelist dict
        node_list: list, a set of nodes to extract their contacts from the graph
        random_selection: bool, wether randomly subsample a set of nodes from graph
        N: int, number of nodes to be randomly sampled from graph
    
    Returns:
        new_edgelist: dict, a dictionary of edges corresponding to nodes in the node_list
    """
    print("Generate graph subsample...")
    if isinstance(graph, dict):
        edgelist = graph
        nodes = node_list(graph)
    else:
        edgelist = graph.edgelist
        nodes = graph.nodes_list()        

    if random_selection:
        node_list = list(np.random.choice(nodes, size = N, replace = False))

    new_edgelist = {}
    for t, edge_data in edgelist.items():
                for (u,v), f in edge_data.items():
                    if u in node_list or v in node_list:
                        if t not in new_edgelist:
                            new_edgelist[t] = {}
                            new_edgelist[t][(u, v)] = f
                        else:
                            new_edgelist[t][(u, v)] = f
    return new_edgelist