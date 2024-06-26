from functools import wraps
from .src.data import inline_export, inline_import, save_optimization_variables, save_all_variables
from .src.graph.variable import Variable, ImplicitVariable, SparseMatrix
from .src.model import Model
from .src.recorder import Recorder
from .src.operations.custom.custom import CustomExplicitOperation
from .src.variable_group import VariableGroup
from .manager import RecManager
from .src.operations.loops.loop import frange
from .utils.parameters import check_parameter
import warnings

manager = RecManager()

def get_current_recorder() -> Recorder:
    if manager.active_recorder is None:
        raise ValueError("No active recorder found. Start a new recorder by csdl.Recorder().start()")
    return manager.active_recorder

def build_new_recorder(inline = False, debug = False, expand_ops = False, auto_hierarchy = False):
    from .src.recorder import Recorder
    return Recorder(inline = inline, debug=debug, expand_ops=expand_ops, auto_hierarchy=auto_hierarchy)


class namespace:
    def __init__(self, namespace: str):
        """Namespacing context manager.

        Parameters
        ----------
        namespace : str
            The name of the namespace to enter.
        """
        self.namespace = namespace
        self.recorder = get_current_recorder()

    def __enter__(self):
        enter_namespace(self.namespace)

    def __exit__(self, exc_type, exc_val, exc_tb):
        exit_namespace()

def enter_namespace(namespace: str):
    """
    Enters a new namespace.

    Args:
        namespace: The name of the namespace to enter.
    """
    recorder = get_current_recorder()
    recorder._enter_namespace(namespace)

def exit_namespace():
    """
    Exits the current namespace.
    """
    recorder = get_current_recorder()
    recorder._exit_namespace()


def enter_subgraph():
    """
    Enters a new subgraph.
    """
    recorder = get_current_recorder()
    recorder._enter_subgraph()

def exit_subgraph():
    """
    Exits the current subgraph.
    """
    recorder = get_current_recorder()
    recorder._exit_subgraph()

def print_all_recorders():
    print(manager)
    

@wraps(Recorder.visualize_graph)
def visualize_graph(*args, **kwargs):
    recorder = get_current_recorder()
    recorder.visualize_graph(*args, **kwargs)