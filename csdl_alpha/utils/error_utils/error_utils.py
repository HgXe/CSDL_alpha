from typing import Union
from csdl_alpha.src.graph.node import Node

def get_check_shape_mismatch_string(a, b, a_str = None, b_str = None):
    """
    get shape mismatch error string. 
    if a_str and b_str are None, it will return a general shape mismatch error string.
    """
    if a_str is None or b_str is None:
        return (f"Shapes do not match: {a.shape} != {b.shape}")
    else:
        return (f"Shape of {a_str} and {b_str} do not match: {a.shape} != {b.shape}")
    
def get_node_name_string(nodes:Union[list[Node], Node]):
    """
    get a string of the nodes in a list
    """
    if isinstance(nodes, Node):
        return nodes.name
    nodes_str = ""
    for node in nodes:
        nodes_str += f"{node.name}, "
    nodes_str = nodes_str[:-2]
    return nodes_str

def check_if_valid_shape(shape):
    """
    checks if shape is a tuple of integers and raises TypeError otherwise
    """
    if not isinstance(shape, tuple):
        raise TypeError(f"shape must be a tuple. {type(shape)} given.")
    for dim in shape:
        import numpy as np
        if not isinstance(dim, (int, np.integer)):
            raise TypeError(f"shape must consist of integers. {type(dim)} given.")
    return True
