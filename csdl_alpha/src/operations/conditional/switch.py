from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple, Union

import numpy as np

from csdl_alpha.src.graph.operation import Operation, set_properties
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import validate_and_variablize, variablize, get_type_string
import csdl_alpha as csdl
import csdl_alpha.utils.testing_utils as csdl_tests


VariableLike = Union[Variable, np.ndarray, float, int]


@set_properties(elementary=False, contains_subgraph=True)
class Switch(Operation):
    """
    Multi-branch control-flow operation that selects one of N branch subgraphs
    to execute based on an integer index, wrapping jax.lax.switch.

    Each branch is defined as a Python callable that takes the same operand
    variables and returns one or more CSDL Variables. All branches must return
    the same number of outputs and matching output shapes.

    Inputs
    ------
    index : Variable (scalar)
        Integer-like selector in [0, num_branches).
    operands : list[Variable]
        The inputs passed to each branch function.
    branches : list[Callable]
        Callables with signature branch(*operands) -> Variable | tuple[Variable,...]
    """

    def __init__(self, index: Variable, operands: Sequence[Variable], branches: Sequence[Callable]):
        # Construct with index and all operand variables as inputs
        super().__init__(index, *operands)
        self.name = "switch"

        if not isinstance(index, Variable):
            raise TypeError(f"index must be a Variable, got {get_type_string(index)}")
        if index.shape != (1,):
            raise ValueError(f"index must be a scalar Variable with shape (1,), got {index.shape}")
        if not isinstance(branches, (list, tuple)) or len(branches) == 0:
            raise TypeError("branches must be a non-empty list/tuple of callables")
        for i, b in enumerate(branches):
            if not callable(b):
                raise TypeError(f"branches[{i}] must be callable, got {get_type_string(b)}")

        self.operands: List[Variable] = list(operands)
        for opnd in self.operands:
            if not isinstance(opnd, Variable):
                raise TypeError("All operands must be Variables")

        self._branch_graphs = []  # list[Graph]
        self._branch_outputs: List[Tuple[Variable, ...]] = []
        self._num_outputs = None
        self._output_shapes: Tuple[tuple, ...] | None = None
        self._jax_branch_fns = None  # lazy-built list of JAX callables

        # Record each branch in its own subgraph, sharing operand variables
        rec = csdl.get_current_recorder()
        for i, branch_fn in enumerate(branches):
            rec._enter_subgraph(add_missing_variables=True, name=f"switch_branch_{i}")
            outs = branch_fn(*self.operands)
            if isinstance(outs, Variable):
                outs_tuple = (outs,)
            elif isinstance(outs, tuple):
                outs_tuple = outs
            else:
                raise TypeError(
                    f"Branch function must return a Variable or tuple of Variables; got {get_type_string(outs)}"
                )

            # Cache outputs and subgraph
            self._branch_outputs.append(outs_tuple)
            branch_graph = rec.active_graph
            rec._exit_subgraph()
            self._branch_graphs.append(branch_graph)

            # Validate consistency of number of outputs and shapes
            if self._num_outputs is None:
                self._num_outputs = len(outs_tuple)
                self._output_shapes = tuple(out.shape for out in outs_tuple)
            else:
                if len(outs_tuple) != self._num_outputs:
                    raise ValueError(
                        f"All branches must return the same number of outputs; expected {self._num_outputs}, got {len(outs_tuple)} for branch {i}"
                    )
                shapes_i = tuple(out.shape for out in outs_tuple)
                if shapes_i != self._output_shapes:
                    raise ValueError(
                        f"All branches must return matching output shapes; expected {self._output_shapes}, got {shapes_i} for branch {i}"
                    )

        # Create outputs for this operation with the agreed shapes
        assert self._output_shapes is not None
        self.set_dense_outputs(self._output_shapes)

    def _set_operand_values(self, *operand_vals):
        # Assign inline values to operand variables so branch subgraphs can execute inline
        for var, val in zip(self.operands, operand_vals):
            var.set_value(val)

    def compute_inline(self, index, *operands):
        # Select branch and execute its subgraph inline
        idx = int(np.asarray(index).reshape(-1)[0])
        if idx < 0 or idx >= len(self._branch_graphs):
            raise IndexError(f"switch index {idx} out of range [0, {len(self._branch_graphs)})")

        # Update operand values
        self._set_operand_values(*operands)

        # Execute only the chosen branch subgraph inline
        self._branch_graphs[idx].execute_inline()
        outs = tuple(out.value for out in self._branch_outputs[idx])
        if len(outs) == 1:
            return outs[0]
        return outs

    def _ensure_jax_branches(self):
        if self._jax_branch_fns is not None:
            return
        from csdl_alpha.backends.jax.graph_to_jax import create_jax_function

        self._jax_branch_fns = []
        for graph, outs in zip(self._branch_graphs, self._branch_outputs):
            jax_fn = create_jax_function(graph, list(outs), self.operands)
            # Ensure tuple output for consistency within lax.switch
            def _wrap(fn):
                return lambda ops: tuple(fn(*ops))

            self._jax_branch_fns.append(_wrap(jax_fn))

    def compute_jax(self, index, *operands):
        # Build JAX branches and dispatch via lax.switch
        import jax.numpy as jnp
        from jax import lax

        self._ensure_jax_branches()
        idx = jnp.asarray(index).astype(jnp.int32)
        ops_tuple = tuple(operands)
        branches = tuple(self._jax_branch_fns)
        return lax.switch(idx, branches, ops_tuple)

    def evaluate_vjp(self, cotangents, index, *operands_and_outputs):
        """Reverse-mode accumulation through the selected branch.

        We treat the index as discrete (zero gradients). Gradients propagate only to
        operands via the active branch subgraph's reverse pass.
        """
        import csdl_alpha as csdl
        from csdl_alpha.src.operations.derivatives.reverse import vjp as vjp_fn
        import numpy as np

        # Split operands and outputs
        operands = operands_and_outputs[: len(self.operands)]
        outputs = operands_and_outputs[len(self.operands) :]

        # Determine wrts (only operands that require gradients)
        wrts = []
        for opnd in operands:
            if cotangents.check(opnd):
                wrts.append(opnd)

        # All the inputs we pass into the composed operation
        output_cotangents = []
        for of in outputs:
            cotangents.initialize(of)
            if cotangents[of] is None:
                cotangents.accumulate(of, csdl.Variable(value = np.zeros(of.shape)))
            output_cotangents.append(cotangents[of])

        # build vjp_fns for all branches
        # the outputs are specific to each branch
        vjp_fns = []
        for branch_graph, branch_outputs in zip(self._branch_graphs, self._branch_outputs):
            def vjp_wrap(*inputs, graph=branch_graph, branch_outputs=branch_outputs):
                # Split inputs into output cotangents and wrts
                num_outputs = len(branch_outputs)
                wrap_output_cotangents = inputs[:num_outputs]
                wrap_wrts = inputs[num_outputs:]
                
                # build seeds for this branch
                seeds = [(var, cot) for (var, cot) in zip(branch_outputs, wrap_output_cotangents)]
                
                # compute VJP for this branch
                vjp_dict = vjp_fn(seeds, wrap_wrts, graph)

                # return wrt cotangents in the same order as wrts
                vjp_outputs = [vjp_dict[wrt] for wrt in wrap_wrts]
                return tuple(vjp_outputs)
            vjp_fns.append(vjp_wrap)

        # Select branch index from current value
        wrt_cots_list = switch(index, vjp_fns, *(list(output_cotangents) + list(wrts)))
        
        # handle single output case
        if isinstance(wrt_cots_list, csdl.Variable):
            wrt_cots_list = (wrt_cots_list,)

        for i, wrt in enumerate(wrts):
            cotangents.accumulate(wrt, wrt_cots_list[i])

        # Also accumulate zero cotangent to index
        if cotangents.check(index):
            zero = csdl.Variable(value=np.zeros(index.shape))
            cotangents.accumulate(index, zero)

def switch(index: VariableLike, branches: Sequence[Callable], *operands: VariableLike) -> Union[Variable, Tuple[Variable, ...]]:
    """
    CSDL multi-branch switch operator wrapping jax.lax.switch.

    Parameters
    ----------
    index : VariableLike (scalar)
        Integer-like selector in [0, len(branches)). When using the JAX backend,
        this may be a float; it will be cast to int32 inside the op.
    branches : Sequence[Callable]
        A list/tuple of callables, each with signature f(*operands) -> Variable or tuple[Variable, ...].
        All branches must return the same number of outputs with the same shapes.
    operands : VariableLike
        Operands passed to each branch function in order.

    Returns
    -------
    Variable | tuple[Variable, ...]
        The outputs from the selected branch.

    Examples
    --------
    >>> import csdl_alpha as csdl
    >>> import numpy as np
    >>> rec = csdl.build_new_recorder(inline=True); rec.start()
    >>> x = csdl.Variable(value=np.array([1.0, 2.0, 3.0]))
    >>> idx = csdl.Variable(value=np.array([1.0]))  # selects branch 1
    >>> def b0(v):
    ...     return v + 1
    >>> def b1(v):
    ...     return 2*v
    >>> y = csdl.switch(idx, [b0, b1], x)
    >>> y.value
    array([2., 4., 6.])
    """
    index_v = validate_and_variablize(index)
    operand_vars = [validate_and_variablize(op) for op in operands]
    op = Switch(index_v, operand_vars, branches)
    return op.finalize_and_return_outputs()


# -----------------------
# Tests
# -----------------------
class TestSwitch(csdl_tests.CSDLTest):
    overwrite_backend = 'jax'  # use JAX backend for gradients and forward eval in tests

    def test_switch_basic(self):
        self.prep(always_build_inline=False)
        import csdl_alpha as csdl
        import numpy as np

        x = csdl.Variable(value=np.array([1.0, 2.0, 3.0]))
        idx = csdl.Variable(value=np.array([0.0]))

        def b0(v):
            return v + 1

        def b1(v):
            return 2 * v

        def b2(v):
            return v - 3

        y = switch(idx, [b0, b1, b2], x)

        compare = []
        compare += [csdl_tests.TestingPair(y, np.array([2.0, 3.0, 4.0]), tag='branch0')]
        self.run_tests(compare_values=compare, verify_derivatives=True)

        # Change index and re-run forward
        idx.value = np.array([2.0])
        csdl.get_current_recorder().active_graph.execute_inline()
        assert np.allclose(y.value, np.array([-2.0, -1.0, 0.0]))

    def test_switch_multi_output(self):
        self.prep(always_build_inline=False)
        import csdl_alpha as csdl
        import numpy as np

        a = csdl.Variable(value=np.array([1.0]))
        b = csdl.Variable(value=np.array([2.0]))
        idx = csdl.Variable(value=np.array([1.0]))

        def br0(x, y):
            s = x + y
            p = x * y
            return s, p

        def br1(x, y):
            s = 2 * x + 3 * y
            p = (x + 1) * (y + 1)
            return s, p

        s, p = switch(idx, [br0, br1], a, b)

        compare = []
        compare += [csdl_tests.TestingPair(s, np.array([8.0]), tag='sum')]
        compare += [csdl_tests.TestingPair(p, np.array([6.0]), tag='prod')]
        self.run_tests(compare_values=compare, verify_derivatives=True)

    def test_shape_mismatch_raises(self):
        self.prep()
        import csdl_alpha as csdl
        import numpy as np

        x = csdl.Variable(value=np.array([1.0, 2.0]))
        i = csdl.Variable(value=np.array([0.0]))

        def b0(v):
            return v

        def b1(v):
            return v[0:1]  # different shape

        try:
            _ = switch(i, [b0, b1], x)
            raise AssertionError("Expected ValueError for shape mismatch")
        except ValueError:
            pass

if __name__ == '__main__':
    test = TestSwitch()
    test.test_switch_basic()
    test.test_switch_multi_output()
    test.test_shape_mismatch_raises()