from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation
from csdl_alpha.src.graph.operation import set_properties 
from csdl_alpha.src.graph.variable import Variable
import csdl_alpha.utils.testing_utils as csdl_tests
import numpy as np
from csdl_alpha.utils.typing import VariableLike
from csdl_alpha.utils.inputs import validate_and_variablize

@set_properties(linear=False)
class Sin(ElementwiseOperation):
    def __init__(self,x):
        super().__init__(x)
        self.name = 'sin'

    def compute_inline(self, x):
        return np.sin(x)

    def evaluate_vjp(self, cotangents, x, y):
        if cotangents.check(x):
            cotangents.accumulate(x, cotangents[y]*cos(x))

@set_properties(linear=False)
class ArcSin(ElementwiseOperation):
    def __init__(self,x):
        super().__init__(x)
        self.name = 'asin'

    def compute_inline(self, x):
        return np.arcsin(x)

    def evaluate_vjp(self, cotangents, x, y):
        if cotangents.check(x):
            cotangents.accumulate(x, cotangents[y]/(1.0 - x**2)**0.5)

@set_properties(linear=False)
class Cos(ElementwiseOperation):
    def __init__(self,x):
        super().__init__(x)
        self.name = 'cos'

    def compute_inline(self, x):
        return np.cos(x)
    
    def evaluate_vjp(self, cotangents, x, y):
        if cotangents.check(x):
            cotangents.accumulate(x, -cotangents[y]*sin(x))

@set_properties(linear=False)
class ArcCos(ElementwiseOperation):
    def __init__(self,x):
        super().__init__(x)
        self.name = 'acos'

    def compute_inline(self, x):
        return np.arccos(x)
    
    def evaluate_vjp(self, cotangents, x, y):
        if cotangents.check(x):
            cotangents.accumulate(x, -cotangents[y]/(1.0 - x**2)**0.5)

@set_properties(linear=False)
class Tan(ElementwiseOperation):
    def __init__(self,x):
        super().__init__(x)
        self.name = 'tan'

    def compute_inline(self, x):
        return np.tan(x)

    def evaluate_vjp(self, cotangents, x, y):
        if cotangents.check(x):
            cotangents.accumulate(x, cotangents[y]/(cos(x)**2))

@set_properties(linear=False)
class ArcTan(ElementwiseOperation):
    def __init__(self,x):
        super().__init__(x)
        self.name = 'atan'

    def compute_inline(self, x):
        return np.arctan(x)

    def evaluate_vjp(self, cotangents, x, y):
        if cotangents.check(x):
            cotangents.accumulate(x, cotangents[y]/(1.0 + x**2))

@set_properties(linear=False)
class Tanh(ElementwiseOperation):
    def __init__(self,x):
        super().__init__(x)
        self.name = 'tanh'

    def compute_inline(self, x):
        return np.tanh(x)

    def evaluate_vjp(self, cotangents, x, y):
        if cotangents.check(x):
            cotangents.accumulate(x, cotangents[y]*(1-tanh(x)**2))

@set_properties(linear=False)
class Sinh(ElementwiseOperation):
    def __init__(self,x):
        super().__init__(x)
        self.name = 'sinh'

    def compute_inline(self, x):
        return np.sinh(x)

    def evaluate_vjp(self, cotangents, x, y):
        if cotangents.check(x):
            cotangents.accumulate(x, cotangents[y]*cosh(x))

@set_properties(linear=False)
class Cosh(ElementwiseOperation):
    def __init__(self,x):
        super().__init__(x)
        self.name = 'cosh'

    def compute_inline(self, x):
        return np.cosh(x)

    def evaluate_vjp(self, cotangents, x, y):
        if cotangents.check(x):
            cotangents.accumulate(x, cotangents[y]*sinh(x))

def sin(x:VariableLike) -> Variable:
    """Elementwise sine of a CSDL Variable

    Parameters
    ----------
    x : Variable
        CSDL Variable to take the sine of

    Returns
    -------
    y: Variable
        The elementwise sine of x
    
    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([1.0, 2.0, 3.0]))
    >>> y = csdl.sin(x)
    >>> y.value
    array([0.84147098, 0.90929743, 0.14112001])

    """
    x = validate_and_variablize(x, raise_on_sparse = False)
    return Sin(x).finalize_and_return_outputs()


def arcsin(x:VariableLike) -> Variable:
    """Elementwise arcsine of a CSDL Variable

    Parameters
    ----------
    x : Variable
        CSDL Variable to take the sine of

    Returns
    -------
    y: Variable
        The elementwise sine of x
    
    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([1.0, -0.5, 0.5]))
    >>> y = csdl.arcsin(x)
    >>> y.value
    array([ 1.57079633, -0.52359878,  0.52359878])
    """
    x = validate_and_variablize(x, raise_on_sparse = False)
    return ArcSin(x).finalize_and_return_outputs()

def sinh(x:VariableLike) -> Variable:
    """Elementwise hyperbolic sine of a CSDL Variable

    Parameters
    ----------
    x : Variable
        CSDL Variable to take the hyperbolic sine of

    Returns
    -------
    y: Variable
        The elementwise hyperbolic sine of x

    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([1.0, 2.0, 3.0]))
    >>> y = csdl.sinh(x)
    >>> y.value
    array([ 1.17520119,  3.62686041, 10.01787493])
    """
    x = validate_and_variablize(x)
    return Sinh(x).finalize_and_return_outputs()

def cos(x:VariableLike) -> Variable:
    """Elementwise cosine of a CSDL Variable

    Parameters
    ----------
    x : Variable
        CSDL Variable to take the cosine of

    Returns
    -------
    y: Variable
        The elementwise cosine of x

        
    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([1.0, 2.0, 3.0]))
    >>> y = csdl.cos(x)
    >>> y.value
    array([ 0.54030231, -0.41614684, -0.9899925 ])
    """
    x = validate_and_variablize(x)
    return Cos(x).finalize_and_return_outputs()

def arccos(x:VariableLike) -> Variable:
    """Elementwise arccosine of a CSDL Variable

    Parameters
    ----------
    x : Variable
        CSDL Variable to take the cosine of

    Returns
    -------
    y: Variable
        The elementwise cosine of x

        
    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([1.0, -0.5, 0.5]))
    >>> y = csdl.arccos(x)
    >>> y.value
    array([0.        , 2.0943951 , 1.04719755])
    """
    x = validate_and_variablize(x)
    return ArcCos(x).finalize_and_return_outputs()

def cosh(x:VariableLike) -> Variable:
    """Elementwise hyperbolic cosine of a CSDL Variable

    Parameters
    ----------
    x : Variable
        CSDL Variable to take the hyperbolic cosine of

    Returns
    -------
    y: Variable
        The elementwise hyperbolic cosine of x

    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([1.0, 2.0, 3.0]))
    >>> y = csdl.cosh(x)
    >>> y.value
    array([ 1.54308063,  3.76219569, 10.067662  ])
    """
    x = validate_and_variablize(x)
    return Cosh(x).finalize_and_return_outputs()

def tan(x:VariableLike) -> Variable:
    """Elementwise tangent of a CSDL Variable

    Parameters
    ----------
    x : Variable
        CSDL Variable to take the tangent of

    Returns
    -------
    y: Variable
        The elementwise tangent of x

    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([1.0, 2.0, 3.0]))
    >>> y = csdl.tan(x)
    >>> y.value
    array([ 1.55740772, -2.18503986, -0.14254654])
    """
    x = validate_and_variablize(x)
    return Tan(x).finalize_and_return_outputs()

def arctan(x:VariableLike) -> Variable:
    """Elementwise arctangent of a CSDL Variable

    Parameters
    ----------
    x : Variable
        CSDL Variable to take the tangent of

    Returns
    -------
    y: Variable
        The elementwise tangent of x

    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([1.0, 2.0, 3.0]))
    >>> y = csdl.arctan(x)
    >>> y.value
    array([0.78539816, 1.10714872, 1.24904577])
    """
    x = validate_and_variablize(x)
    return ArcTan(x).finalize_and_return_outputs()

def tanh(x:VariableLike) -> Variable:
    """Elementwise hyperbolic tangent of a CSDL Variable

    Parameters
    ----------
    x : Variable
        CSDL Variable to take the hyperbolic tangent of

    Returns
    -------
    y: Variable
        The elementwise hyperbolic tangent of x

    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([1.0, 2.0, 3.0]))
    >>> y = csdl.tanh(x)
    >>> y.value
    array([0.76159416, 0.96402758, 0.99505475])
    """
    x = validate_and_variablize(x)
    return Tanh(x).finalize_and_return_outputs()

class TestTrig(csdl_tests.CSDLTest):
    
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np
        x_val = 3.0
        y_val = np.arange(10).reshape(2,5)
        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)

        compare_values = []
        # sin/cos/tan scalar variables
        s1 = csdl.sin(x)
        t1 = np.sin(x_val).flatten()
        compare_values += [csdl_tests.TestingPair(s1, t1)]

        s2 = csdl.cos(x)
        t2 = np.cos(x_val).flatten()
        compare_values += [csdl_tests.TestingPair(s2, t2)]

        s3 = csdl.tan(x)
        t3 = np.tan(x_val).flatten()
        compare_values += [csdl_tests.TestingPair(s3, t3)]

        # compare_values = []
        # sin/cos/tan scalar variables
        s1a = csdl.arcsin(x)
        t1a = np.arcsin(x_val).flatten()
        compare_values += [csdl_tests.TestingPair(s1a, t1a)]

        s2a = csdl.arccos(x)
        t2a = np.arccos(x_val).flatten()
        compare_values += [csdl_tests.TestingPair(s2a, t2a)]

        s3a = csdl.arctan(x)
        t3a = np.arctan(x_val).flatten()
        compare_values += [csdl_tests.TestingPair(s3a, t3a)]

        # sin/cos/tan tensor variables
        s4 = csdl.sin(y)
        t4 = np.sin(y_val)
        compare_values += [csdl_tests.TestingPair(s4, t4)]

        s5 = csdl.cos(y)
        t5 = np.cos(y_val)
        compare_values += [csdl_tests.TestingPair(s5, t5)]

        s6 = csdl.tan(y)
        t6 = np.tan(y_val)
        compare_values += [csdl_tests.TestingPair(s6, t6)]

        # sin/cos/tan tensor variables
        s4a = csdl.arcsin(y)
        t4a = np.arcsin(y_val)
        compare_values += [csdl_tests.TestingPair(s4a, t4a)]

        s5a = csdl.arccos(y)
        t5a = np.arccos(y_val)
        compare_values += [csdl_tests.TestingPair(s5a, t5a)]

        s6a = csdl.arctan(y)
        t6a = np.arctan(y_val)
        compare_values += [csdl_tests.TestingPair(s6a, t6a)]

        s7a = csdl.tanh(y)
        t7a = np.tanh(y_val)
        compare_values += [csdl_tests.TestingPair(s7a, t7a)]

        s8a = csdl.sinh(y)
        t8a = np.sinh(y_val)
        compare_values += [csdl_tests.TestingPair(s8a, t8a)]

        s9a = csdl.cosh(y)
        t9a = np.cosh(y_val)
        compare_values += [csdl_tests.TestingPair(s9a, t9a)]

        self.run_tests(compare_values = compare_values, verify_derivatives=True)

    def test_examples(self):
        self.docstest(sin)
        self.docstest(cos)
        self.docstest(tan)
        self.docstest(arcsin)
        self.docstest(arccos)
        self.docstest(arctan)
        self.docstest(tanh)
        self.docstest(sinh)
        self.docstest(cosh)
