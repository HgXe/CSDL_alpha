from csdl_alpha.src.operations.implicit_operations.nonlinear_solvers.nonlinear_solver import NonlinearSolver, check_variable_shape_compatibility, check_run_time_tolerance
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.typing import VariableLike
import numpy as np
import math
from typing import Union, Callable

class BracketedSearch(NonlinearSolver):
    
    def __init__(
            self,
            name:str = 'bracketed_search',
            print_status:bool = True,
            tolerance:VariableLike=1e-10,
            max_iter:int=100,
            elementwise_states=False,
            residual_jac_kwargs = None,
        ):
        """Initialize the BracketedSearch solver.

        Parameters
        ----------
        name : str, optional
            The name of the solver. Defaults to 'bracketed_search'.
        print_status : bool, optional
            Whether to print the solver status during execution. Defaults to True.
        tolerance : VariableLike, optional
            The tolerance value used to determine convergence. Defaults to 1e-10.
        max_iter : int, optional
            The maximum number of iterations allowed. Defaults to 100.
        """

        super().__init__(
            name = name,
            print_status = print_status,
            tolerance = tolerance,
            max_iter = max_iter,
            elementwise_states = elementwise_states,
            residual_jac_kwargs = residual_jac_kwargs,
        )

        self.add_metadata('tolerance', tolerance)
        self.add_metadata('max_iter', max_iter)

    def add_state(
            self,
            state: Variable,
            residual: Variable,
            bracket: tuple[VariableLike, VariableLike],
            tolerance: VariableLike = None
        ):
        """Add a state to the solver.

        This method adds a state to the solver along with its corresponding residual. It also allows specifying a bracket
        and tolerance for the state.

        Parameters
        ----------
        state : Variable
            The state variable to be added to the solver.
        residual : Variable
            The residual variable corresponding to the state.
        bracket : tuple[Union[Variable, np.ndarray, float, int], Union[Variable, np.ndarray, float, int]]
            The bracket for the state. It should be a tuple containing two elements, each of which can be a Variable,
            np.ndarray, float, or int. Must have a shape of (1,) or match the shape of the state.
        tolerance : Union[Variable, np.ndarray, float], optional
            The tolerance for the state. It can be a Variable, np.ndarray, float, or int. If not provided, the tolerance
            from the solver's metadata will be used. Must have a shape of (1,) or match the shape of the state.

        Raises
        ------
        ValueError
            If the bracket is not a tuple or does not have two elements.
        ValueError
            If the bracket elements have invalid types or shapes.
        ValueError
            If the tolerance has an invalid type or shape.
        """

        self.add_state_residual_pair(state, residual)

        # check if bracket is valid and store it
        if not isinstance(bracket, tuple):
            raise TypeError(f"Bracket must be a tuple. Got {type(bracket)}")
        if len(bracket) != 2:    
            raise ValueError(f"Bracket must have two elements. Got {len(bracket)}")
        for element in bracket:
            if isinstance(element, (Variable, np.ndarray)):
                if not (element.shape == (1,) or element.shape == state.shape):
                    raise ValueError(f"Bracket shape must match state shape. {element.shape} given")
            elif not isinstance(element, (float, int, np.integer)):
                raise TypeError(f"Invalid bracket element type, got {type(element)}")
        self.add_state_metadata(state, 'bracket_l', bracket[0])
        self.add_state_metadata(state, 'bracket_r', bracket[1])

        # check if tolerance is valid and store it
        self.add_tolerance(state, tolerance)

    def _inline_solve_(self):
        iter = 0

        # I'm going to assemble everything into a big vector, solve it, and then unpack it

        # First, I need to know the size of the vector, and the indices of each state
        indices_dict = {}
        size = 0
        for state in self.state_to_residual_map.keys():
            state_size = math.prod(state.shape)
            indices_dict[state] = slice(size, size+state_size)
            size += state_size

        x_upper = np.empty(size)
        x_lower = np.empty(size)
        r_lower = np.empty(size)
        r_update = np.empty(size)
        tolerance = np.empty(size)

        def compute(x, r):
            for state, index in indices_dict.items():
                state.value = x[index].reshape(state.shape)
            self.update_residual()
            for state, index in indices_dict.items():
                r[index] = self.state_to_residual_map[state].value.flatten()

        def to_array(x):
            if isinstance(x, Variable):
                return x.value
            elif isinstance(x, np.ndarray):
                return x
            elif isinstance(x, (float, int)):
                return np.array([x])

        # Now, I need to populate the x vectors with the initial bracket values
        # and populate the tolerance vector
        for state in self.state_to_residual_map.keys():
            bracket = (self.state_metadata[state]['bracket_l'], self.state_metadata[state]['bracket_r'])
            ith_tolerance = self.state_metadata[state]['tolerance']
            # bracket indices could be Variable or ndarray of shape (1,) or state.shape
            # or float or int
            # I want to turn them all into numpy arrays
            
            # now I can flatten them and put them in the x vectors
            x_lower[indices_dict[state]] = to_array(bracket[0]).flatten()
            x_upper[indices_dict[state]] = to_array(bracket[1]).flatten()
            tolerance[indices_dict[state]] = to_array(ith_tolerance).flatten()

        # Now, I need to populate the r vectors with the residuals of the initial bracket values
        # I'm also going to record whether the sign of the lower is negative (True) or positive (False)
        compute(x_lower, r_lower)
        # compute(x_upper, r_upper)
        r_sign = r_lower < 0

        # Now, I need to iterate until the bracket is small enough
        while True:
            # compute the midpoint
            x_mid = (x_upper + x_lower) / 2
            compute(x_mid, r_update)

            # check if the midpoint is the new upper or lower bound
            lower_mask = (r_update < 0) == r_sign

            x_lower[lower_mask] = x_mid[lower_mask]
            x_upper[~lower_mask] = x_mid[~lower_mask]

            # check if the bracket is small enough
            if check_run_time_tolerance(r_update, tolerance):
                converged = True
                break

            # check if we've hit the max iterations
            iter += 1
            if iter > self.metadata['max_iter']:
                converged = False
                break

        if self.print_status:
            print(self._inline_print_nl_status(iter, converged))
        

    def _jax_solve_(
            self,
            jax_residual_function:Callable,
            jax_intermediate_function:Callable,
            input_dict:dict):
        import jax
        import jax.numpy as jnp
        import jax.lax as lax
        # I'm going to assemble everything into a big vector, solve it, and then unpack it

        # First, I need to know the size of the vector, and the indices of each state
        indices_dict = {}
        size = 0
        for state in self.state_to_residual_map.keys():
            state_size = state.size
            indices_dict[state] = slice(size, size+state_size)
            size += state_size

        x_upper = jnp.empty(size)
        x_lower = jnp.empty(size)
        r_update = jnp.ones(size) # need this to start greater than tolerance
        tolerance = jnp.empty(size)

        def compute(x):
            states = []
            for state, index in indices_dict.items():
                states.append(x[index].reshape(state.shape))
            output_residuals = jax_residual_function(states)
            res_array = jnp.empty(size)
            for i, ind in enumerate(indices_dict.values()):
                res_array = res_array.at[ind].set(output_residuals[i].flatten())
            return res_array

        def to_array(x):
            if isinstance(x, Variable):
                try:
                    return input_dict[x]
                except KeyError:
                    # NOTE: If it's not in the input_vars, it must have a constant value (I think at least)
                    return x.value
            elif isinstance(x, (np.ndarray, jnp.ndarray)):
                return x
            elif isinstance(x, (float, int)):
                return jnp.array([x])

        # Now, I need to populate the x vectors with the initial bracket values
        # and populate the tolerance vector
        for state in self.state_to_residual_map.keys():
            bracket = (self.state_metadata[state]['bracket_l'], self.state_metadata[state]['bracket_r'])
            ith_tolerance = self.state_metadata[state]['tolerance']
            # bracket indices could be Variable or ndarray of shape (1,) or state.shape
            # or float or int
            # I want to turn them all into jax arrays
            
            # now I can flatten them and put them in the x vectors
            x_lower = x_lower.at[indices_dict[state]].set(to_array(bracket[0]).flatten())
            x_upper = x_upper.at[indices_dict[state]].set(to_array(bracket[1]).flatten())
            tolerance = tolerance.at[indices_dict[state]].set(to_array(ith_tolerance).flatten())

        # Now, I need to populate the r vectors with the residuals of the initial bracket values
        # I'm also going to record whether the sign of the lower is negative (True) or positive (False)
        r_lower = compute(x_lower)
        # compute(x_upper, r_upper)
        r_sign = r_lower < 0

        # Now, I need to iterate until the bracket is small enough
        def loop_body(val): # x_lower, x_upper, r_update, i
            x_lower = val[0]
            x_upper = val[1]
            i = val[3]
            i += 1
            x_mid = (x_upper + x_lower) / 2
            r_update = compute(x_mid)
            lower_mask = (r_update < 0) == r_sign
            x_lower = lax.select(lower_mask, x_mid, x_lower)
            x_upper = lax.select(~lower_mask, x_mid, x_upper)
            return (x_lower, x_upper, r_update, i)

        def loop_cond(val): # x_lower, x_upper, r_update, i
            r_update = val[2]
            i = val[3]
            return ~(jnp.all(jnp.less_equal(jnp.abs(r_update), tolerance)) | jnp.greater_equal(i, self.metadata['max_iter']))
        
        x_lower, x_upper, r_update, i = lax.while_loop(loop_cond, loop_body, (x_lower, x_upper, r_update, 0))
        x_final = (x_upper + x_lower) / 2 # might as well
        states = []
        for state, index in indices_dict.items():
            states.append(x_final[index].reshape(state.shape))
        return states

