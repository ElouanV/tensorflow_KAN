import tensorflow as tf

import numpy as np
from sklearn.linear_model import LinearRegression
import sympy

# sigmoid = sympy.Function('sigmoid')
# name: (tf implementation, sympy implementation)
SYMBOLIC_LIB = {'x': (lambda x: x, lambda x: x),
                 'x^2': (lambda x: x**2, lambda x: x**2),
                 'x^3': (lambda x: x**3, lambda x: x**3),
                 'x^4': (lambda x: x**4, lambda x: x**4),
                 '1/x': (lambda x: 1/x, lambda x: 1/x),
                 '1/x^2': (lambda x: 1/x**2, lambda x: 1/x**2),
                 '1/x^3': (lambda x: 1/x**3, lambda x: 1/x**3),
                 '1/x^4': (lambda x: 1/x**4, lambda x: 1/x**4),
                 'sqrt': (lambda x: tf.sqrt(x), lambda x: sympy.sqrt(x)),
                 '1/sqrt(x)': (lambda x: 1/tf.sqrt(x), lambda x: 1/sympy.sqrt(x)),
                 'exp': (lambda x: tf.exp(x), lambda x: sympy.exp(x)),
                 'log': (lambda x: tf.log(x), lambda x: sympy.log(x)),
                 'abs': (lambda x: tf.abs(x), lambda x: sympy.Abs(x)),
                 'sin': (lambda x: tf.sin(x), lambda x: sympy.sin(x)),
                 'tan': (lambda x: tf.tan(x), lambda x: sympy.tan(x)),
                 'tanh': (lambda x: tf.tanh(x), lambda x: sympy.tanh(x)),
                 'sigmoid': (lambda x: tf.sigmoid(x), sympy.Function('sigmoid')),
                 #'relu': (lambda x: tf.relu(x), relu),
                 'sgn': (lambda x: tf.sign(x), lambda x: sympy.sign(x)),
                 'arcsin': (lambda x: tf.arcsin(x), lambda x: sympy.arcsin(x)),
                 'arctan': (lambda x: tf.arctan(x), lambda x: sympy.atan(x)),
                 'arctanh': (lambda x: tf.arctanh(x), lambda x: sympy.atanh(x)),
                 '0': (lambda x: x*0, lambda x: x*0),
                 'gaussian': (lambda x: tf.exp(-x**2), lambda x: sympy.exp(-x**2)),
                 'cosh': (lambda x: tf.cosh(x), lambda x: sympy.cosh(x)),
                 #'logcosh': (lambda x: tf.log(tf.cosh(x)), lambda x: sympy.log(sympy.cosh(x))),
                 #'cosh^2': (lambda x: tf.cosh(x)**2, lambda x: sympy.cosh(x)**2),
}

def add_symbolic(name, fun):
    '''
    add a symbolic function to library
    
    Args:
    -----
        name : str
            name of the function
        fun : fun
            torch function or lambda function
    
    Returns:
    --------
        None
    
    Example
    -------
    >>> print(SYMBOLIC_LIB['Bessel'])
    KeyError: 'Bessel'
    >>> add_symbolic('Bessel', torch.special.bessel_j0)
    >>> print(SYMBOLIC_LIB['Bessel'])
    (<built-in function special_bessel_j0>, Bessel)
    '''
    exec(f"globals()['{name}'] = sympy.Function('{name}')")
    SYMBOLIC_LIB[name] = (fun, globals()[name])