from torch import Tensor
from typing import List, Optional, Union


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()

def subgraph(subset: Union[Tensor, List[int]], edge_index: Tensor, edge_attr: Optional[Tensor] = None, relabel_nodes: bool = False, num_nodes: Optional[int] = None):

    device = edge_index.device
    node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    node_mask[subset] = 1

    if relabel_nodes:
        node_idx = torch.zeros(num_nodes, dtype=torch.long, device=device)
        node_idx[subset] = torch.arange(subset.size(0), device=device)

    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    edge_index = node_idx[edge_index]
    return edge_index, edge_attr
