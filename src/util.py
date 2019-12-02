from __future__ import division

'''Various utility functions (from pyetc)'''

import sys, os, re
import six
import numpy as np
from numpy import nan, inf, NaN, Inf
import pprint
from collections import defaultdict, OrderedDict
import ast

#Define the location of DATADIR here, so it can be imported
#and used by other modules in the pyetc package.
DATADIR = os.path.join(os.path.split(__file__)[0],'data')
INST_DATADIR = None # will be set by etc_instruments package

TESTDIR = os.path.join(os.path.split(__file__)[0],'test')

# Limitations of pysynphot make it unable to accept file paths
# which contain a "-".  Permitting a "-",  e.g. site-packages,
# will break the nonstellar source.
assert "-" not in DATADIR, "DATADIR must not have a dash, but does: %s" % DATADIR

#..........................................................


def stable_dict(dictionary=None):
    ''' Returns a dictionary of the appropriate type.

        Under python 2, dictionary iterators are stable
        in between successivea calls within the same process,
        or from process to procees. Under python 3, dictionary
        iterators return elements in random order, different
        from call to call. This function returns an OrderedDict
        instance that works under both 2 and 3, but leaves open
        the possibility to use other kinds of dictionary if the
        need ever arises.

        This function should be used to create dictionary instances
        throughout the pyetc code, instead of direct calls to dict().
    '''
    if dictionary:
        return OrderedDict([item for item in six.iteritems(dictionary)])
    else:
        return OrderedDict()


def read_dict(fname):
    '''read a python dictionary from a file that was written
    with write_dict.
    '''
    # no six for this
    if sys.version_info[0] >= 3:
        f=open(fname,'r', encoding="utf-8")
    else:
        f=open(fname,'r')
    datastr = f.read()
    f.close()
    # convert DOS file to Unix - otherwise the eval will fail
    datastr = datastr.replace('\r','')
    try :
        datadict = safe_eval(datastr)
    except Exception as e:
        print('EXCEPTION:',e)
        print('cannot eval data in file ',fname)
        raise
    return datadict


def write_dict(datadict, fname, header=None):
    '''write a python dictionary to a file in such a way that it
    can be read with read_dict.
    At present, this means using pprint, but encapsulating it
    here will make it easier if we decide to change the format as the
    project evolves.
    A string, or a list of strings, passed in the optional header keyword
    will be prepended with a # sign before being written to the file,
    which read_dict will ignore. This allows the files to be
    documented.
    '''

    fh=open(fname,'w')
    #First write commented-out header
    if header is not None:
        #Support either a list or a string
        if isinstance(header,list):
            for line in header:
                fh.write("# %s\n"%line)
        else:
            fh.write("# %s\n"%header)
    #Now write the data
    pprint.pprint(datadict,fh)
    fh.close()

def combine_dicts(flist,nodup=False):
    """ Read one dict per file into a master dictionary.
    The return dictionary will be structured as
    key:[list of values]. By default, duplicate values will
    remain in place; if nodup=True, they will be removed
    without affecting order."""

    master=defaultdict(list)

    for fname in flist:
        d=read_dict(fname)
        #If nodup evaluates to True, check for duplicates
        for key, val in list(d.items()):
            if nodup and val in master[key]:
                pass
            else:
                master[key].append(val)

    return master


def flatten(ndict, depth):
    '''assumes data is a nested dict of specified depth

    return a list of dicts (flattened) and a list of tuples
    with the indices that select that item'''
    retlist = []
    keyset = []
    keys = list(ndict.keys())
    for key in keys:
        value = ndict[key]
        if depth == 1:
            retlist.append(value)
            keyset.append((key,))
        else:
            newrlist, newkeyset = flatten(value,depth-1)
            retlist = retlist + newrlist
            # need to prepend the existing key to those returned
            newkeyset = [(key,) + item for item in newkeyset]
            keyset = keyset + newkeyset
    return retlist, keyset

def bilinear_interp(xref, yref, arr, x, y):
    '''perform bilinear interpolation on 2-d array.

    xref: contains x values corresponding to first dim index in array.
    yref: contains y values corresponding to second dim index in array.
    arr: array of values to be interpolated.
    x, y: x and y values for interpolation, scalars or arrays permitted.

    Poor error handling at the moment...
    '''

    xind = np.searchsorted(xref, x)
    yind = np.searchsorted(yref, y)

    # Bound the indices to 0 .. N-1
    xind0 = np.maximum(0, xind - 1)
    yind0 = np.maximum(0, yind - 1)
    xind1 = np.minimum(len(xref) - 1, xind)
    yind1 = np.minimum(len(yref) - 1, yind)

    # The denominator may be zero if outside of the bounds of the
    # image
    xden = (xref[xind1] - xref[xind0])
    xden = np.where(xden == 0.0, 1.0, xden)
    yden = (yref[yind1] - yref[yind0])
    yden = np.where(yden == 0.0, 1.0, yden)

    xfract = (x - xref[xind0]) / xden
    xfracti = 1.0 - xfract
    yfract = (y - yref[yind0]) / yden
    yfracti = 1.0 - yfract

    val = (arr[xind0,yind0] * xfracti * yfracti +
           arr[xind1,yind0] * xfract * yfracti +
           arr[xind0,yind1] * xfracti * yfract +
           arr[xind1,yind1] * xfract * yfract)
    return val

def linear_interp(xref, arr, x):
    '''perform linear interpolation on 1-d array.

    xref: contains x values corresponding to index in array.

    arr: array of values to be interpolated.
    x: x values for interpolation, scalars or arrays permitted.

    works on interpolating rows out of 2-d arrays as well.

    Poor error handling at the moment...
    '''
    xind = np.searchsorted(xref, x)
    xind0 = np.maximum(0, xind - 1)
    xind1 = np.minimum(len(xref) - 1, xind)
    xden = (xref[xind1] - xref[xind0])
    xden = np.where(xden == 0.0, 1.0, xden)
    xfract = (x - xref[xind0]) / xden
    val = arr[xind0] * (1.0 - xfract) + arr[xind1] * xfract
    return val

def comma2num(numstr):
    '''take comma-ed numbers and return appropriate numeric type'''
    # strip commas
    try:
        numstr = numstr.replace(',','')
    except AttributeError:
        #If it has no .replace, it's not a string. Must already
        #be a number. return as is.
        return numstr

    #Now return int or float
    try:
        val = int(numstr)
        return val
    except ValueError:
        #int('3.2') will raise a ValueError, even though int(3.2) does not.
        #Must be a float.
        return float(numstr)

def almostequal(a,b,tol=0.01):
    if a == 0 or b == 0:
        return False
    else:
        if a > b:
            ratio = b/a
        else:
            ratio = a/b
        return np.abs(ratio-1.) < tol

def debug_array(wave, flux, refwave, size=10):
    # Debug helper function. This function prints the input array's contents
    # at the neighborhood of the given wavelength. It comes in handy when
    # comparing python arrays with the contents of the JETC blackboard dump.
    # This method is necessary because np arrays' contents are not accessible
    # from the Wing debugger.
    # wave    -  array with wavelengths
    # flux    -  array with corresponding fluxes
    # refwave -  wavelength value at the midpoint of the printout
    # size    -  how many array elements on the printout
    deltawave = wave - refwave
    index = np.where(np.abs(deltawave) == np.abs(deltawave).min())[0][0]
    for i in range(index-size/2,index+size/2):
        if i >= 0 and i < len(wave):
            print(wave[i], "   ", flux[i])



# Basically permit dotted identifiers, not worrying about invalid
# package specifiers or what is being imported, but ensuring that
# nothing more exotic can possibly be exec'ed.
PACKAGE_RE = re.compile("[A-Za-z_0-9.]+")

def dynamic_import(package):
    """imports a module specified by string `package` which is
    not known until runtime.

    Returns a module/package.

    The purpose of this function is to concentrate a number of uses
    of the "exec" statement in once place where it can be policed
    for security reasons.
    """
    if not PACKAGE_RE.match(package):
        raise ImportError("Invalid dynamic import " + repr(package))

    # six.exec_ doesn't support keyword arguments, even though the
    # documentation says it does.
    six.exec_("import " + package + " as pkg_name", None, locals())

    m = locals()['pkg_name']

    return m

# Taken from Python-2.7 and enhanced to support np.nan and np.inf
def safe_eval(node_or_string):
    """
    Safely evaluate an expression node or a string containing a Python
    expression.  The string or node provided may only consist of the following
    Python literal structures: strings, numbers, tuples, lists, dicts, booleans,
    and None.
    """
    _safe_names = {'None': None, 'True': True, 'False': False,
                   'nan':nan, 'NaN':np.NaN,
                   'inf':inf, 'Inf':np.Inf}
    if isinstance(node_or_string, six.string_types):
        node_or_string = ast.parse(node_or_string, mode='eval')
    if isinstance(node_or_string, ast.Expression):
        node_or_string = node_or_string.body
    def _convert(node):
        # Old code used to return instances of ast.Dict.
        # New code sometimes retuns dicts disguised as lists
        # of tuples (actuaally, ast.List and ast.Tuple), as
        # the first element of the 'args' attribute of an
        # ast.Call instace.
        if isinstance(node, ast.Call):
            result = OrderedDict()
            if len(node.args) > 0:
                for element in node.args[0].elts:
                    name  = element.elts[0].s
                    v = element.elts[1]
                    if isinstance(v, ast.Str):
                        value = v.s
                    elif isinstance(v, ast.Num):
                        value = v.n
                    else:
                        value = None
                    result[name] = value
            return result
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Tuple):
            return tuple(map(_convert, node.elts))
        elif isinstance(node, ast.List):
            return list(map(_convert, node.elts))
        elif isinstance(node, ast.Dict):
            return OrderedDict((_convert(k), _convert(v)) for k, v
                         in zip(node.keys, node.values))
        elif isinstance(node, ast.Name):
            if node.id in _safe_names:
                return _safe_names[node.id]
        elif isinstance(node, ast.BinOp) and \
             isinstance(node.op, (ast.Add, ast.Sub)) and \
             isinstance(node.right, ast.Num) and \
             isinstance(node.right.n, complex) and \
             isinstance(node.left, ast.Num) and \
             isinstance(node.left.n, six.integer_types) and \
             isinstance(node.left.n,  float):
            left = node.left.n
            right = node.right.n
            if isinstance(node.op, ast.Add):
                return left + right
            else:
                return left - right

        # when the expression contains a negative number,
        # under python 3 the negative number will be cast
        # as an instance of UnaryOp with first operand of
        # type USub.
        elif isinstance(node, ast.NameConstant):
            if sys.version_info[0] >= 3:
                return node.value
        elif isinstance(node, ast.UnaryOp) and \
             isinstance(node.op, ast.USub) and \
             isinstance(node.operand, ast.Num):
            return -(node.operand.n)

        raise ValueError('malformed string')

    return _convert(node_or_string)


# The simpler code:
#
# return sum(collection.rate(rebinned, qyc) for k in pieces)
#
# doesn't work in general because the sum() function initializer
# (as in sum(...., initializer) should be set to an object of
# correct type. This calculation has to be performed over scalar
# floats, numpy float arrays, and zeroed SpectralVector instances
# with attributes matching the SpectralVector instances returned
# by the .rate() method when applied to a collection. Building this
# initializer object requires awkward code. In the implementations
# below, we resort to (also) awkward but easier to read code.
#
# Code snippets that call sum() in circumstances similar to these
# are interspersed over the engine code base. Mostly they appear in
# extracted.py and cross_collection.py, but might be found elsewhere
# as well. We just have no way at this point to find which ones are
# prone to fail and thus have to be replaced by calls to the functions
# below. The code that calls these sum() instances is never executed
# within the realm of the pyetc main algorithms as laid out in module
# calc_functions.py.

def sum_rate_over_collection(collection, rebinned, qyc):
    first = True
    result = None   # to avoid a code analysis flag....
    for k in collection:
        if first:
            result = k.rate(rebinned, qyc)
            first = False
        else:
            result += k.rate(rebinned, qyc)

    return result

def sum_counts_over_collection(collection, read_pattern, per_read, rebinned, qyc):
    first = True
    result = None   # to avoid a code analysis flag....
    for k in collection:
        if first:
            result = k.counts(read_pattern, per_read, rebinned, qyc)
            first = False
        else:
            result += k.counts(read_pattern, per_read, rebinned, qyc)

    return result



