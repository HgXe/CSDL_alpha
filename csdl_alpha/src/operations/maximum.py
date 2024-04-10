from csdl_alpha.src.graph.operation import Operation, set_properties
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize
import csdl_alpha.utils.test_utils as csdl_tests
import csdl_alpha as csdl

import numpy as np

class Maximum(Operation):
    '''
    Maximum entries in the input tensor along the specified axes.
    '''
    def __init__(self, x, axes=None, out_shape=None, rho=20.):
        super().__init__(x)
        self.name  = 'maximum'
        out_shapes = (out_shape,)
        self.axes  = axes
        self.rho   = rho
        self.set_dense_outputs(out_shapes)
        in_shape = x.shape

        if axes is not None:
            axes = self.axes  = tuple(np.sort(axes))
            rank = len(in_shape)
            alphabet  = 'abcdefghijklmnopqrstuvwxyz'
            in1_str = alphabet[:axes[0]]
            in2_str = alphabet[axes[0]]
            ones_shape = (in_shape[axes[0]],)
            for i in range(len(axes)-1):
                in1_str += alphabet[axes[i] + 1 : axes[i + 1]]
                in2_str += alphabet[axes[i+1]]
                ones_shape += (in_shape[axes[i+1]],)
            in1_str += alphabet[axes[-1] + 1 : rank]
            self.einsum_str = '{},{}->{}'.format(
                in1_str,
                in2_str,
                alphabet[:rank],
                )
            self.ones_shape = ones_shape

    def compute_inline(self, x):
        rho = self.rho
        axes = self.axes

        if axes is None:
            x_max = np.max(x)
            smooth_max = x_max + 1/rho * np.log(np.sum(np.exp(rho * (x - x_max))))
            return smooth_max
        else:
            ones_shape = self.ones_shape
            axeswise_max = np.max(x, axis=self.axes)
            # print(self.einsum_str, axeswise_max.shape, ones_shape)
            difference = x - np.einsum(
                self.einsum_str,
                axeswise_max,
                np.ones(ones_shape),
                )
            exp = np.exp(rho * difference)
            summation = np.sum(exp, axis=axes)
            smooth_axeswise_max = axeswise_max + 1.0 / rho * np.log(summation)
            return smooth_axeswise_max

class ElementwiseMaximum(Operation):
    '''
    Elementwise maximum of all the Variables in the arguments.
    '''
    def __init__(self, *args, rho=20.):
        super().__init__(*args)
        self.name = 'elementwise_maximum'
        out_shapes = (args[0].shape,)
        self.rho   = rho
        self.set_dense_outputs(out_shapes)

    def compute_inline(self, *args):
        rho = self.rho
        ew_max = args[0]
        for arg in args[1:]:
            ew_max = np.maximum(ew_max, arg)

        summation = 0.
        for arg in args:
            summation += np.exp(rho * (arg - ew_max))

        smooth_ew_max = (ew_max + 1. / rho * np.log(summation))
        return smooth_ew_max

def maximum(*args, axes=None, rho=20.):
    """
    doc strings
    """
    # Multiple Variables to find maximum
    if axes is not None and len(args) > 1:
        raise ValueError('Cannot find maximum of multiple Variables along specified axes. \
                         Use X = max(A,B,...) followed by out=max(X, axes=(...)) instead.')
    if any(args[i].shape != args[0].shape for i in range(1, len(args))):
        raise ValueError('All Variables must have the same shape.')
    
    # Single Variable to find maximum
    if axes is not None:
        if any(np.asarray(axes) > (len(args[0].shape)-1)):
            raise ValueError('Specified axes cannot be more than the rank of the Variable.')
        if any(np.asarray(axes) < 0):
            raise ValueError('Axes cannot have negative entries.')

    if len(args) == 1:
        if axes is None:
            out_shape = (1,)
        else:
            out_shape = tuple([x for i, x in enumerate(args[0].shape) if i not in axes])
            if len(out_shape) == 0:
                raise ValueError('Cannot find maximum of a scalar Variable along all axes. \
                                 Use maximum(A) to find the maximum of a tensor Variable.')
        
        op = Maximum(variablize(args[0]), axes=axes, out_shape=out_shape, rho=rho)
    else:
        # axes is None for multiple variables
        args = [variablize(x) for x in args]
        op = ElementwiseMaximum(*args, rho=rho)
    
    return op.finalize_and_return_outputs()

class TestMaximum(csdl_tests.CSDLTest):
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()
        x_val = 3.0*np.arange(6).reshape(2,3)
        y_val = 2.0*np.ones((2,3))
        z_val = np.ones((2,3))
        d_val = np.arange(12).reshape(2,3,2)


        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)
        z = csdl.Variable(name = 'z', value = z_val)
        d = csdl.Variable(name = 'd', value = d_val)

        compare_values = []
        # maximum of a single tensor variable
        s1 = csdl.maximum(x)
        t1 = np.array([15.0])
        compare_values += [csdl_tests.TestingPair(s1, t1, tag = 's1')]

        # maximum of a single tensor constant
        s2 = csdl.maximum(x_val)
        compare_values += [csdl_tests.TestingPair(s2, t1, tag = 's2')]

        # maximum of a single tensor variable along specified axes
        s3 = csdl.maximum(x, axes=(1,))
        t3 = np.max(x_val, axis=1)
        compare_values += [csdl_tests.TestingPair(s3, t3, tag = 's3')]

        # maximum of a single tensor variable along 2 specified axes
        s4 = csdl.maximum(d, axes=(0,2))
        t4 = np.max(d_val, axis=(0,2))
        compare_values += [csdl_tests.TestingPair(s4, t4, tag = 's4', decimal=8)]

        # elementwise maximum of multiple tensor variables
        s5 = csdl.maximum(x, y, z)
        t5 = np.maximum(x_val, y_val)
        compare_values += [csdl_tests.TestingPair(s5, t5, tag = 's5', decimal=8)]

        # elementwise maximum of multiple tensor constants
        s6 = csdl.maximum(x_val, y_val, z_val)
        compare_values += [csdl_tests.TestingPair(s6, t5, tag = 's6', decimal=8)]

        self.run_tests(compare_values = compare_values,)


    def test_example(self,):
        self.prep()

        # docs:entry
        import csdl_alpha as csdl
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()
        x_val = 3.0*np.arange(6).reshape(2,3)
        y_val = 2.0*np.ones((2,3))
        z_val = np.ones((2,3))
        d_val = np.arange(12).reshape(2,3,2)

        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)
        z = csdl.Variable(name = 'z', value = z_val)
        d = csdl.Variable(name = 'd', value = d_val)

        # maximum of a single tensor variable
        s1 = csdl.maximum(x)
        print(s1.value)

        # maximum of a single tensor constant
        s2 = csdl.maximum(x_val)
        print(s2.value)

        # maximum of a single tensor variable along a specified axis
        s3 = csdl.maximum(x, axes=(1,))
        print(s3.value)

        # maximum of a single tensor variable along 2 specified axes
        s4 = csdl.maximum(d, axes=(0,2))

        # maximum of multiple tensor variables
        s5 = csdl.maximum(x, y, z)
        print(s5.value)

        # maximum of multiple tensor constants and variables
        s6 = csdl.maximum(x_val, y_val, z)
        print(s6.value)
        # docs:exit

        compare_values = []
        t1 = np.array([15.])
        t3 = np.max(x_val, axis=1)
        t4 = np.max(d_val, axis=(0,2))
        t5 = np.maximum(x_val, y_val)

        compare_values += [csdl_tests.TestingPair(s1, t1, tag = 's1')]
        compare_values += [csdl_tests.TestingPair(s2, t1, tag = 's2')]
        compare_values += [csdl_tests.TestingPair(s3, t3, tag = 's3')]
        compare_values += [csdl_tests.TestingPair(s4, t4, tag = 's4', decimal=8)]
        compare_values += [csdl_tests.TestingPair(s5, t5, tag = 's5', decimal=8)]
        compare_values += [csdl_tests.TestingPair(s6, t5, tag = 's6', decimal=8)]
        
        self.run_tests(compare_values = compare_values,)

if __name__ == '__main__':
    test = TestMaximum()
    test.test_functionality()
    test.test_example()