"""PyStan utility functions

These functions validate and organize data passed to and from the
classes and functions defined in the file `stan_fit.hpp` and wrapped
by the Cython file `stan_fit.pxd`.

"""
#-----------------------------------------------------------------------------
# Copyright (c) 2013-2015, PyStan developers
#
# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
#-----------------------------------------------------------------------------

# REF: rstan/rstan/R/misc.R
from numbers import Number
import numpy as np
import re
from collections import OrderedDict
import sys
PY2 = sys.version_info[0] == 2
_identity = lambda x: x

if PY2:
    from collections import Callable, Sequence
else:
    from collections.abc import Callable, Sequence

string_types = (str,)


def is_legal_stan_vname(name):
    stan_kw1 = ('for', 'in', 'while', 'repeat', 'until', 'if', 'then', 'else',
                'true', 'false')
    stan_kw2 = ('int', 'real', 'vector', 'simplex', 'ordered', 'positive_ordered',
                'row_vector', 'matrix', 'corr_matrix', 'cov_matrix', 'lower', 'upper')
    stan_kw3 = ('model', 'data', 'parameters', 'quantities', 'transformed', 'generated')
    cpp_kw = ("alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor", "bool",
              "break", "case", "catch", "char", "char16_t", "char32_t", "class", "compl",
              "const", "constexpr", "const_cast", "continue", "decltype", "default", "delete",
              "do", "double", "dynamic_cast", "else", "enum", "explicit", "export", "extern",
              "false", "float", "for", "friend", "goto", "if", "inline", "int", "long", "mutable",
              "namespace", "new", "noexcept", "not", "not_eq", "nullptr", "operator", "or", "or_eq",
              "private", "protected", "public", "register", "reinterpret_cast", "return",
              "short", "signed", "sizeof", "static", "static_assert", "static_cast", "struct",
              "switch", "template", "this", "thread_local", "throw", "true", "try", "typedef",
              "typeid", "typename", "union", "unsigned", "using", "virtual", "void", "volatile",
              "wchar_t", "while", "xor", "xor_eq")
    illegal = stan_kw1 + stan_kw2 + stan_kw3 + cpp_kw
    if re.findall(r'(\.|^[0-9]|__$)', name):
        return False
    return not name in illegal


def _dict_to_rdump(data):
    parts = []
    for name, value in data.items():
        if isinstance(value, (Sequence, Number, np.number, np.ndarray, int, bool, float)) \
           and not isinstance(value, string_types):
            value = np.asarray(value)
        else:
            raise ValueError("Variable {} is not a number and cannot be dumped.".format(name))

        if value.dtype == np.bool:
            value = value.astype(int)

        if value.ndim == 0:
            s = '{} <- {}\n'.format(name, str(value))
        elif value.ndim == 1:
            s = '{} <-\nc({})\n'.format(name, ', '.join(str(v) for v in value))
        elif value.ndim > 1:
            tmpl = '{} <-\nstructure(c({}), .Dim = c({}))\n'
            # transpose value as R uses column-major
            # 'F' = Fortran, column-major
            s = tmpl.format(name,
                            ', '.join(str(v) for v in value.flatten(order='F')),
                            ', '.join(str(v) for v in value.shape))
        parts.append(s)
    return ''.join(parts)


def stan_rdump(data, filename):
    """
    Dump a dictionary with model data into a file using the R dump format that
    Stan supports.

    Parameters
    ----------
    data : dict
    filename : str

    """
    for name in data:
        if not is_legal_stan_vname(name):
            raise ValueError("Variable name {} is not allowed in Stan".format(name))
    with open(filename, 'w') as f:
        f.write(_dict_to_rdump(data))


def _rdump_value_to_numpy(s):
    """
    Convert a R dump formatted value to Numpy equivalent

    For example, "c(1, 2)" becomes ``array([1, 2])``

    Only supports a few R data structures. Will not work with European decimal format.
    """
    if "structure" in s:
        vector_str, shape_str = re.findall(r'c\([^\)]+\)', s)
        shape = [int(d) for d in shape_str[2:-1].split(',')]
        if '.' in vector_str:
            arr = np.array([float(v) for v in vector_str[2:-1].split(',')])
        else:
            arr = np.array([int(v) for v in vector_str[2:-1].split(',')])
        # 'F' = Fortran, column-major
        arr = arr.reshape(shape, order='F')
    elif "c(" in s:
        if '.' in s:
            arr = np.array([float(v) for v in s[2:-1].split(',')], order='F')
        else:
            arr = np.array([int(v) for v in s[2:-1].split(',')], order='F')
    else:
        arr = np.array(float(s) if '.' in s else int(s))
    return arr


def _remove_empty_pars(pars, pars_oi, dims_oi):
    """
    Remove parameters that are actually empty. For example, the parameter
    y would be removed with the following model code:

        transformed data { int n; n <- 0; }
        parameters { real y[n]; }

    Parameters
    ----------
    pars: iterable of str
    pars_oi: list of str
    dims_oi: list of list of int

    Returns
    -------
    pars_trimmed: list of str
    """
    pars = list(pars)
    for par, dim in zip(pars_oi, dims_oi):
        if par in pars and np.prod(dim) == 0:
            del pars[pars.index(par)]
    return pars


def read_rdump(filename):
    """
    Read data formatted using the R dump format

    Parameters
    ----------
    filename: str

    Returns
    -------
    data : OrderedDict
    """
    contents = open(filename).read().strip()
    names = [name.strip() for name in re.findall(r'^(\w+) <-', contents, re.MULTILINE)]
    values = [value.strip() for value in re.split('\w+ +<-', contents) if value]
    if len(values) != len(names):
        raise ValueError("Unable to read file. Unable to pair variable name with value.")
    d = OrderedDict()
    for name, value in zip(names, values):
        d[name.strip()] = _rdump_value_to_numpy(value.strip())
    return d
