import numpy as np

def ingest_value(value):
    if isinstance(value, (float, int)):
        value = np.array([value])
    elif not isinstance(value, np.ndarray) and value is not None:
        raise ValueError(f"Value must be a numpy array, float or int. {value} given")
    return value

def scalarize(value):
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value[0]
        else:
            raise ValueError(f"Value must be a scalar. {value.shape} given")
    elif not isinstance(value, (float, int)):
        raise ValueError(f"Value must be a scalar. {value} given")
    else:
        return value

def variablize(variable):
    from csdl_alpha.src.graph.variable import Variable
    if isinstance(variable, Variable):
        return variable
    else:
        var = Variable(value = ingest_value(variable))
        return var

def get_shape(shape, value):
    if shape is None:
        if value is not None:
            shape = value.shape
        else:
            raise ValueError("Shape or value must be provided")
    else:
        if value is not None:
            if shape != value.shape:
                raise ValueError("Shape and value shape must match")
    return shape