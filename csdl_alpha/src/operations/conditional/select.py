from csdl_alpha.src.graph.operation import Operation, set_properties
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize, validate_and_variablize
import csdl_alpha.utils.testing_utils as csdl_tests
from csdl_alpha.utils.typing import VariableLike
import csdl_alpha as csdl
from typing import Optional, Callable
import numpy as np

@set_properties()
class Select(Operation):
    '''
    Selects elements from two arrays based on a condition tensor.
    This operation wraps jax.numpy.where (which internally uses jax.lax.select).
    
    Parameters:
    -----------
    pred : Variable
        A boolean tensor or tensor of 0s and 1s that determines which elements to select
    on_true : Variable
        Values to select when pred is True/1
    on_false : Variable  
        Values to select when pred is False/0
        
    Returns:
    --------
    Variable
        A tensor with the same shape as the inputs, containing elements from on_true 
        where pred is True/1, and elements from on_false where pred is False/0
    '''

    def __init__(self, pred: Variable, on_true: Variable, on_false: Variable, logic: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> Variable:
        super().__init__(pred, on_true, on_false)
        self.name = 'select'
        self.logic = logic  # Optional custom logic function for selection
        
        # Validate inputs
        if not isinstance(pred, Variable):
            raise TypeError("pred must be a Variable")
        if not isinstance(on_true, Variable):
            raise TypeError("on_true must be a Variable") 
        if not isinstance(on_false, Variable):
            raise TypeError("on_false must be a Variable")
            
        # Check shape compatibility
        if on_true.shape != on_false.shape:
            raise ValueError(f"on_true and on_false must have the same shape, got {on_true.shape} and {on_false.shape}")
            
        # pred can be broadcasted to the shape of on_true/on_false
        try:
            np.broadcast_shapes(pred.shape, on_true.shape)
        except ValueError:
            raise ValueError(f"pred shape {pred.shape} cannot be broadcast to on_true/on_false shape {on_true.shape}")
        
        # Output has the same shape as on_true/on_false
        out_shapes = (on_true.shape,)
        self.set_dense_outputs(out_shapes)

    def compute_inline(self, pred, on_true, on_false):
        if self.logic is not None:
            pred = self.logic(pred)
        # Use numpy.where for inline computation
        return np.where(pred, on_true, on_false)
    
    def compute_jax(self, pred, on_true, on_false):
        import jax
        import jax.numpy as jnp
        if self.logic is not None:
            # Run Python logic (expects numpy arrays) via pure_callback
            # Ensures bool dtype and same shape as pred
            # def logic_wrapper(x):
            #     x_np = np.asarray(x)
            #     out = self.logic(x_np)
            #     return np.asarray(out, dtype=bool)
            pred_bool = jax.pure_callback(
                self.logic,
                jax.ShapeDtypeStruct(pred.shape, jnp.bool_),
                pred,
            )
        else:
            pred_bool = jnp.asarray(pred, dtype=bool)
        return jnp.where(pred_bool, on_true, on_false)

    def evaluate_vjp(self, cotangents, pred, on_true, on_false, y):
        """
        Evaluate vector-Jacobian product (reverse mode AD)
        """
        import csdl_alpha as csdl
        
        if cotangents.check(on_true):
            # Gradient flows to on_true where pred is True
            # Create zeros with the same shape as cotangents[y]
            zeros_like_cotangent = csdl.Variable(shape=cotangents[y].shape, value=0.0)
            cotangents.accumulate(on_true, csdl.select(pred, cotangents[y], zeros_like_cotangent, logic=self.logic))
            
        if cotangents.check(on_false):
            # Gradient flows to on_false where pred is False
            # Create zeros with the same shape as cotangents[y]
            zeros_like_cotangent = csdl.Variable(shape=cotangents[y].shape, value=0.0)
            cotangents.accumulate(on_false, csdl.select(pred, zeros_like_cotangent, cotangents[y], logic=self.logic))

        # pred is boolean/discrete, so no gradient flows through it


def select(pred: VariableLike, on_true: VariableLike, on_false: VariableLike, logic: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> Variable:
    """
    Selects elements from two arrays based on a condition tensor.
    This function wraps jax.numpy.where (which internally uses jax.lax.select).
    
    Parameters:
    -----------
    pred : VariableLike
        A boolean tensor or tensor of 0s and 1s that determines which elements to select
    on_true : VariableLike
        Values to select when pred is True/1
    on_false : VariableLike  
        Values to select when pred is False/0
    logic : Optional[Callable[[np.ndarray], np.ndarray]]
        A function that takes a numpy array and returns a numpy array. This function is applied to the pred tensor before the selection is made.

    Returns:
    --------
    Variable
        A tensor with the same shape as the inputs, containing elements from on_true 
        where pred is True/1, and elements from on_false where pred is False/0
        
    Examples:
    ---------
    >>> import csdl_alpha as csdl
    >>> pred = csdl.Variable(value=np.array([True, False, True]))
    >>> on_true = csdl.Variable(value=np.array([1.0, 2.0, 3.0]))
    >>> on_false = csdl.Variable(value=np.array([4.0, 5.0, 6.0]))
    >>> result = csdl.select(pred, on_true, on_false)
    # result.value = [1.0, 5.0, 3.0]
    """
    pred = validate_and_variablize(pred)
    on_true = validate_and_variablize(on_true) 
    on_false = validate_and_variablize(on_false)

    op = Select(pred, on_true, on_false, logic=logic)
    return op.finalize_and_return_outputs()


@set_properties()
class SelectN(Operation):
    '''
    Selects elements from N arrays based on a condition tensor.
    This operation wraps jax.lax.select_n.
    
    Parameters:
    -----------
    pred : Variable
        A boolean tensor or tensor of 0s and 1s that determines which elements to select
    *args : Variable
        Values to select from, where the i-th argument corresponds to the i-th value in pred
        
    Returns:
    --------
    Variable
        A tensor with the same shape as the inputs, containing elements from the i-th argument
        where pred is i
    '''

    def __init__(self, pred: Variable, *args: Variable) -> Variable:
        super().__init__(pred, *args)
        self.name = 'select_n'

        # pred can be broadcasted to the shape of all other inputs
        # for arg in args:
        #     try:
        #         np.broadcast_shapes(pred.shape, arg.shape)
        #     except ValueError:
        #         raise ValueError(f"pred shape {pred.shape} cannot be broadcast to {arg.shape}")

        # Output has the same shape as all other inputs
        out_shapes = (args[0].shape,)
        self.set_dense_outputs(out_shapes)

    def compute_inline(self, pred, *args):
        # Use numpy.where for inline computation
        result = np.empty_like(args[0])
        for i in range(len(args)):
            mask = (pred == i)
            result[mask] = args[i][mask]
        return result
        
    def compute_jax(self, pred, *args):
        import jax.lax as jlax
        import jax.numpy as jnp
        pred_int = jnp.asarray(pred, dtype=int)
        return jlax.select_n(pred_int, *args)

    def evaluate_vjp(self, cotangents, pred, *args_and_y):
        """
        Evaluate vector-Jacobian product (reverse mode AD)
        """
        import csdl_alpha as csdl
        *args, y = args_and_y
        
        for i, arg in enumerate(args):
            if cotangents.check(arg):
                # Create zeros with the same shape as cotangents[y]
                # zeros_like_cotangent = cotangents[y] * 0.0
                zeros_like_cotangent = csdl.Variable(shape=cotangents[y].shape, value=0.0)
                # cot_args = [cotangents[y] if j == i else zeros_like_cotangent for j in range(len(args))]
                logic = lambda x, i=i: x == i
                cotangents.accumulate(arg, csdl.select(pred, cotangents[y], zeros_like_cotangent, logic=logic))

        # pred is boolean/discrete, so no gradient flows through it - set to zero
        if cotangents.check(pred):
            cotangents.accumulate(pred, csdl.Variable(shape=pred.shape, value=0.0))

def select_n(pred: VariableLike, *args: VariableLike) -> Variable:
    """
    Selects elements from two arrays based on a condition tensor.
    This function wraps jax.numpy.where (which internally uses jax.lax.select).
    
    Parameters:
    -----------
    pred : VariableLike
        A boolean tensor or tensor of 0s and 1s that determines which elements to select
    *args : VariableLike
        Values to select from, where the i-th argument corresponds to the i-th value in pred

    Returns:
    --------
    Variable
        A tensor with the same shape as the inputs, containing elements from the i-th argument
        where pred is i
        
    Returns:
    --------
    Variable
        A tensor with the same shape as the inputs, containing elements from on_true 
        where pred is True/1, and elements from on_false where pred is False/0
        
    Examples:
    ---------
    >>> import csdl_alpha as csdl
    >>> pred = csdl.Variable(value=np.array([0, 1, 2]))
    >>> on_0 = csdl.Variable(value=np.array([1.0, 2.0, 3.0]))
    >>> on_1 = csdl.Variable(value=np.array([4.0, 5.0, 6.0]))
    >>> on_2 = csdl.Variable(value=np.array([7.0, 8.0, 9.0]))
    >>> result = csdl.select(pred, on_0, on_1, on_2)
    # result.value = [1.0, 5.0, 9.0]
    """
    pred = validate_and_variablize(pred)
    args = [validate_and_variablize(arg) for arg in args]

    # check that pred and args are all the same shape
    for arg in args:
        if arg.shape != pred.shape:
            raise ValueError(f"All args must have the same shape as pred, got {arg.shape} and {pred.shape}")

    op = SelectN(pred, *args)
    return op.finalize_and_return_outputs()

# Tests
class TestSelect(csdl_tests.CSDLTest):
    
    def test_select_basic(self,):
        recorder = csdl.build_new_recorder(inline=True)
        recorder.start()
        
        # Test basic select operation
        pred_val = np.array([True, False, True, False])
        on_true_val = np.array([1.0, 2.0, 3.0, 4.0])
        on_false_val = np.array([10.0, 20.0, 30.0, 40.0])
        
        pred = csdl.Variable(value=pred_val)
        on_true = csdl.Variable(value=on_true_val)
        on_false = csdl.Variable(value=on_false_val)
        
        result = select(pred, on_true, on_false)
        
        recorder.stop()
        
        expected = np.where(pred_val, on_true_val, on_false_val)
        np.testing.assert_array_equal(result.value, expected)
        
    def test_select_broadcasting(self,):
        recorder = csdl.build_new_recorder(inline=True)
        recorder.start()
        
        # Test broadcasting
        pred_val = np.array([True, False])
        on_true_val = np.array([[1.0, 2.0], [3.0, 4.0]])
        on_false_val = np.array([[10.0, 20.0], [30.0, 40.0]])
        
        pred = csdl.Variable(value=pred_val)
        on_true = csdl.Variable(value=on_true_val)
        on_false = csdl.Variable(value=on_false_val)
        
        result = select(pred, on_true, on_false)
        
        recorder.stop()
        
        # The broadcasting should work the same as numpy.where - element-wise
        expected = np.where(pred_val, on_true_val, on_false_val)
        np.testing.assert_array_equal(result.value, expected)
        
    def test_select_numeric_pred(self,):
        recorder = csdl.build_new_recorder(inline=True)
        recorder.start()
        
        # Test with numeric predicate (0s and 1s)
        pred_val = np.array([1, 0, 1, 0])
        on_true_val = np.array([1.0, 2.0, 3.0, 4.0])
        on_false_val = np.array([10.0, 20.0, 30.0, 40.0])
        
        pred = csdl.Variable(value=pred_val)
        on_true = csdl.Variable(value=on_true_val)
        on_false = csdl.Variable(value=on_false_val)
        
        result = select(pred, on_true, on_false)
        
        recorder.stop()
        
        expected = np.where(pred_val, on_true_val, on_false_val)
        np.testing.assert_array_equal(result.value, expected)

    def test_select_derivatives(self,):
        recorder = csdl.build_new_recorder(inline=False)
        recorder.start()
        
        pred_val = np.array([True, False, True])
        on_true_val = np.array([1.0, 2.0, 3.0])
        on_false_val = np.array([4.0, 5.0, 6.0])
        
        pred = csdl.Variable(value=pred_val)
        on_true = csdl.Variable(value=on_true_val)
        on_false = csdl.Variable(value=on_false_val)
        
        result = select(pred, on_true, on_false)
        
        # Compute derivatives before stopping recorder
        derivative_on_true = csdl.derivative(result, on_true)
        derivative_on_false = csdl.derivative(result, on_false)
        
        recorder.stop()
        
        # Test derivatives using simulator
        from csdl_alpha.backends.simulator import PySimulator
        sim = PySimulator(recorder)
        sim.run()
        
        expected_result = np.array([1.0, 5.0, 3.0])
        np.testing.assert_array_equal(sim[result], expected_result)
        
        expected_deriv_true = np.diag([1.0, 0.0, 1.0])  # gradient flows where pred is True
        np.testing.assert_array_equal(sim[derivative_on_true], expected_deriv_true)
        
        # Check derivative w.r.t. on_false  
        expected_deriv_false = np.diag([0.0, 1.0, 0.0])  # gradient flows where pred is False
        np.testing.assert_array_equal(sim[derivative_on_false], expected_deriv_false)

    def test_select_with_logic(self,):
        recorder = csdl.build_new_recorder(inline=True)
        recorder.start()
        pred_val = np.array([-1.0, 0.5, 2.0, -0.2])
        on_true_val = np.array([1., 2., 3., 4.])
        on_false_val = np.array([10., 20., 30., 40.])
        pred = csdl.Variable(value=pred_val)
        on_true = csdl.Variable(value=on_true_val)
        on_false = csdl.Variable(value=on_false_val)

        # custom logic: positive entries select on_true
        def logic(x):
            return x > 0.0

        op = Select(pred, on_true, on_false, logic=logic)
        result, = op.finalize_and_return_outputs()
        recorder.stop()

        expected = np.where(pred_val > 0.0, on_true_val, on_false_val)
        np.testing.assert_array_equal(result.value, expected)


if __name__ == '__main__':
    # Run basic test
    import pytest
    pytest.main([__file__])