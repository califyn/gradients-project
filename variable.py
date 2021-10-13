import math
import warnings
import numpy as np

def is_float(inn):
    """check if the input is float, or castable to float"""

    try:
        float(inn)
        return True
    except:
        return False

def safe_log(val, max_l=20):
    """a logarithm function that returns max_l for tiny or huge numbers instead of err"""

    try:
        return math.log(val)
    except:
        if val < 0.01 and 0 <= val:
            return -max_l
        elif val > 100000:
            return max_l
        else:
            raise ValueError("cannot safe log " + str(val) + "?")

def safe_inv(val, min_l=0.001):
    """a reciprocal function that is bounded above by 1/min_l, reducing errs"""

    if isinstance(val, Variable):
        val = val()
    elif isinstance(val, Array):
        val = val()

    if val >= min_l:
        return 1 / val
    elif abs(val) < min_l:
            return 1 / min_l
    else:
        raise ValueError("cannot safe invert " + str(val) + "?")

class _CallableArray():
    """
    a list wrapper that can be called, non-accessable otherwise.
    for syntactical consistency.

    users should not be using this type; instead, call this object
    to release a list, before doing anything else with it.
    """

    def __init__(self, arr):
        """initialize object"""

        self.arr = arr

    def __call__(self):
        """when the array is called, it calls all of its elements."""

        return [x() for x in self.arr]

class Gradient():
    """
    custom gradient object to handle partial derivative indexing.
    this object stores the primitive Variables for which the
    parent Variable is a function of, as well as a function
    that can release the value of the partial derivative when
    called.

    this object also supports limited operations with Variables 
    to allow construction of custom operations if necessary. 

    this object should only be accessed through its parent 
    Variable and only by indexing with other Variables.  only
    access the partial derivatives to immediately evaluate them;
    they do not update.

    :param vars: the primitive Variables in the gradient.
    :param exprs: functions that evaluate the corresponding
        partial derivative. this list shares indices with
        vars.
    """

    def __init__(self, vars, exprs):
        """initialize object"""

        self.vars = vars
        self.exprs = exprs

    def __getitem__(self, idx):
        """
        index the gradient with Variables or Arrays to get
        a partial derivative. note that the returned Variable
        must be called in order to release the actual values.
        immediately call the obtained Array after indexing,
        as it will no longer receive updates.

        :param idx: Variable or Array that the parent Variable
            is being differetianted from
        :return: Variable or Array containing partial derivative
            values
        """

        if isinstance(idx, Variable):
            if idx._primitive:
                if idx in self.vars:
                    val = self.exprs[self.vars.index(idx)]()
                    if not is_float(val):
                        raise ValueError("gradient vals should be floats... oops")

                    ret = Variable(expr=val)
                    return ret
                else:
                    warnings.warn("Taking partial derivative wrt a primitive (" + str(idx) + ") not in the def of var")
                    return Variable(expr=0)
            else:
                warnings.warn("Taking partial derivative wrt a non-primitive var is not yet tested")
                
                common_prims = list(set(self.vars).intersection(idx.grad.vars))

                if len(common_prims) == 0:
                    warnings.warn("the derivative here is zero as there are no primitives in common")

                def sum_grad():
                    ret = 0
                    for var in common_prims:
                        ret = ret + self[var]() / other.grad[var]()
                    return ret

                return sum_grad
        elif isinstance(idx, Array):
            ret = []
            for it, x in enumerate(idx.vars):
                ret.append(self[x]())
            return Array(ret)
        else:
            try:
                ret = []
                for x in idx:
                    ret.append(self[x])
                return Array(ret)
            except:
                raise ValueError("can only index gradient by iterable or single Variable")

    def __add__(self, other):
        """
        handle Gradient addition.

        :raises ValueError: when adding to something other than
            a Gradient.  in normal calculus this should not
            have to happen, so it is not supported.
        """

        if isinstance(other, Gradient):
            combined_vars = list(set(self.vars + other.vars))
            combined_exprs = []
            
            def combined_grad(var): # lambda weirdness requires this
                return lambda: self.exprs[self.vars.index(var)]() + other.exprs[other.vars.index(var)]()

            for it, var in enumerate(combined_vars):
                if var in self.vars and var in other.vars:
                    combined_exprs.append(combined_grad(var))
                elif var in self.vars and var not in other.vars:
                    combined_exprs.append(self.exprs[self.vars.index(var)])
                elif var not in self.vars and var in other.vars:
                    combined_exprs.append(other.exprs[other.vars.index(var)])

            return Gradient(combined_vars, combined_exprs)
        else:
            raise ValueError("gradients can only be added to other gradients")

    def __sub__(self, other):
        """handles Gradient subtraction"""

        return self + (-other)

    def __mul__(self, other):
        """handles Gradient multiplication.

        :raises ValueError: when multipling by something that is
            neither a float, nor a Variable.  Gradients should 
            not have to be multiplied by other Gradients in normal
            calculus, so it is not supported.
        """

        if is_float(other):
            other = float(other)

            exprs = []

            def lambda_generator(expr): # lambda weirdness requires this
                expr_ = expr
                other_ = other
                return lambda: expr_() * other_

            for expr in self.exprs:
                exprs.append(lambda_generator(expr))
            
            return Gradient(self.vars, exprs)
        elif isinstance(other, Variable):
            exprs = []

            def lambda_generator(expr): # lambda weirdess requires this
                return lambda: expr() * other()

            for it, expr in enumerate(self.exprs):
                l = lambda_generator(expr)
                exprs.append(lambda_generator(expr))

            return Gradient(self.vars, exprs)
        else:
            raise ValueError("gradients can only be multiplied with floats")
    
    def __truediv__(self, other):
        """handle Gradient division

        :raises ValueError: when dividing by a non-float.
            Division by Variables or Gradients should not happen.
        """

        if is_float(other):
            other = float(other)
            def div_other(inn): # lambda weirdness requires this
                return lambda: inn() / other

            return Gradient(self.vars, [div_other(expr) for expr in self.exprs])
        else:
            raise ValueError("gradients can only be divided by floats")

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return -(self - other)

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * (-1)

class Variable():
    """
    variable class with a bunch of bells and whistles.  features:
        - autodiff--get the partial derivative dy/dx with x.grad[y]()
        - you can set variables to actual values and never have to worry
          about them again
        - computation is run only when the variable or its gradient
          are called, so everything updates in real time & you can set
          variables to be empty to initialize them later
        - safe reassignment to actually change the values of variables
          without making a whole new object. plus, the ability to check
          for circular definitions.

    this makes the syntax much more like actual python, barring the calls
    everywhere; i.e. no accessing gradients by specifying what the function
    values are.
    
    call the variable to release its value; e.g. x() gives the value of x.

    :param _primitive: whether the variable is primitive (e.g. a root of the
        computation graph).
    :param _override: whether this variable is a gradient-less variable.  DO
        NOT USE GRADIENT-LESS VARIABLES, they should only be used in internal
        computations.
    """

    nsy_ok = False # see not_set_yet for clarification

    def not_set_yet():
        """
        error message if a variable is not set yet.  this is turned off
        when silently evaluating during checking for circular definitions,
        otherwise it should be on (mediated by nsy_ok).
        """

        if not Variable.nsy_ok:
            raise ValueError("this primitive Variable has not been set yet")
        else:
            return 0
    
    def overriden_grad():
        """error message for gradients of gradient-less variables"""

        raise ValueError("tried to take gradient of a gradient-less variable")

    def __init__(self, expr=None, grad=None, _override=False):
        """initialize object.

        the variable is inferred to be primitive if expr is of type float.

        :param expr: the value of the variable, float or function
        :param grad: Gradient object to associate with this variable
        :param _override: specify a gradient-less variable.  DO NOT USE.
        """

        is_primitive_def = is_float(expr) or (expr == None)

        # check for bad definitions
        if is_primitive_def and grad != None:
            raise ValueError("primitive Variables cannot have gradients")
        if (not is_primitive_def) and grad == None:
            if not _override:
                raise ValueError("non-primitive Variables must have gradients")
        if _override and (grad != None or is_primitive_def):
            raise ValueError("never use _override with a real gradient or with primitive expr")

        # initialize params
        self._primitive = is_primitive_def
        self._override = _override

        if self._override:
            self.eval = expr
            self.grad = Variable.overriden_grad
        elif self._primitive:
            if expr == None:
                self.eval = Variable.not_set_yet
            else:
                self.eval = lambda: expr
            self.grad = Gradient(vars=[self], exprs=[lambda: 1])
        else:
            self.eval = expr
            self.grad = grad

    def set(self, val, primitive=False):
        """
        safe reassignments.  when modifying the value of any variable,
        use this method (do not do x = ....; that will delete the old object.)

        :param val: the value to set it to.
        :param primitive: whether the value should be interpreted as a 
            primitive variable/float.
        """

        if self._override:
            raise ValueError("cannot change value of a gradient-less variable")

        if is_float(val):
            if not self._primitive:
                raise ValueError("only primitive Variables can be assigned to floats")

            self.eval = lambda: float(val)
        elif isinstance(val, Variable) and primitive:
            if not self._primitive:
                raise ValueError("only primitive Variables can be assigned to other primitive values")

            val = val()
            self.eval = lambda: val
        elif isinstance(val, Variable) and not primitive:
            # check for circular assignments

            def cyclical_assignment():
                raise ValueError("cyclical assignment")

            self.eval = cyclical_assignment
            
            Variable.nsy_ok = True # shut off not-set-yet error message
            val()
            Variable.nsy_ok = False

            self.eval = val.eval
            self.grad = val.grad
        else:
            raise ValueError("cannot set Variables to anything other than float or Variable")

    def clip(self, max_abs):
        """clipping for large values to [-max_abs, max_abs]"""

        if not self._primitive:
            raise ValueError("do not clip non primitive vars")
        
        val = self()
        if abs(val) > max_abs:
            if val < 0:
                self.set(-1 * max_abs)
            else:
                self.set(max_abs)

    def __call__(self):
        """allow the variable to be called"""

        return self.eval()

    @staticmethod
    def set_vars(vars, vals):
        """allow batch setting variables"""

        for (var, val) in zip(vars, vals):
            var.set(val)
        return

    def __add__(self, other):
        """variable addition"""

        if is_float(other):
            other = float(other)
            return Variable(
                expr=lambda: self() + other,
                grad=self.grad
            )
        elif isinstance(other, Variable):
            return Variable(
                expr=lambda: self() + other(),
                grad=self.grad + other.grad
            )
        else:
            raise ValueError("Variables can only be added to floats or other Variables")

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        """variable multiplication"""

        if is_float(other):
            other = float(other)
            return Variable(
                expr=lambda: self() * other,
                grad= self.grad * other
            )
        elif isinstance(other, Variable):
            return Variable(
                expr=lambda: self() * other(),
                grad= self.grad * other + other.grad * self
            )
        elif isinstance(other, Gradient):
            return other * self
        else:
            raise ValueError("Variables can only be multiplied with floats or other Variables")

    def __truediv__(self, other):
        """variable division"""

        if is_float(other):
            other = float(other)
            return Variable(
                expr=lambda: self() * safe_inv(other),
                grad=self.grad * other
            )
        elif isinstance(other, Variable):
            return self * (1 / other)
        else:
            raise ValueError("Variables can only be divided by floats or other Variables")

    def __pow__(self, other):
        """
        variable exponentiation. 

        note that the class does not support expressions
        of the form x^y where x and y are *both* variables.
        there is not enough of a use case for this to justify
        implementing this
        """

        if is_float(other):
            other = float(other)
            return Variable(
                expr=lambda: self() ** other,
                grad=(self.grad * Variable(expr=lambda: self() ** (other - 1), _override=True)) * other
            )
        else:
            raise ValueError("Variables can only be raised to float powers")

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return -(self - other)

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        if is_float(other):
            other = float(other)
            return Variable(
                expr=lambda: other  * safe_inv(self()),
                grad=(self.grad * Variable(expr=lambda: - safe_inv(self())**2, _override=True)) * other
            )
        elif isinstance(other, Variable):
            return other.__truediv__(self)
        else:
            raise ValueError("Variables can only be divided by floats or other Variables")

    def __rpow__(self, other):
        if is_float(other):
            other = float(other)
            return Variable(
                expr=lambda: other ** self(),
                grad=(self.grad * Variable(expr=lambda: other ** self(), _override=True)) * safe_log(other) 
            )
        else:
            raise ValueError("Variables can only be the exponent of a float")

    def __neg__(self):
        return self * (-1)

class ArrayGradient():
    """object to allow batch gradients for arrays"""

    def __init__(self, arr):
        self.arr = arr

    def __getitem__(self, var):
        return _CallableArray([x.grad[var] for x in self.arr])

class Array():
    """
    an array wrapper for lists of variables which supports a few
    basic operations.  internally does not use numpy so may be
    a bit slow, but is generally compatible with numpy ndarrays.
    only supports one-dimensional arrays.

    :param vars: the variables inside the array. can be accessed
        with normal list indexing.
    """

    def __init__(self, arr_expr=None, size=10, primitive=False):
        """
        initialize object

        behavior depends on type of arr_expr:
            - if arr_expr is already an array, we make a copy of
              that array
            - if arr_expr is tuple/list/np.ndarray, if the values
              are all floats we make them primitive Variables,
              else we just make the Array the list of Variables
            - if arr_expr is a function that should be evaluated,
              we evaluate it :param:size times to fill the array.

        :param arr_expr: source for array initialization.
        :param size: specified array size, use only for expression
            based generation.
        :param primitive: whether or not the Variables in the
            Array should be primitive. (if yes, the expression
            in arr_expr is evaluated immediately.)
        """

        if isinstance(arr_expr, Array):
            if not primitive:
                self.vars = arr_expr.vars
            else:
                self.vars = [Variable(expr=x()) for x in arr_expr.vars]
        elif isinstance(arr_expr, tuple) or isinstance(arr_expr, list) or isinstance(arr_expr, np.ndarray):
            if isinstance(arr_expr, list):
                arr = tuple(arr_expr)
            elif isinstance(arr_expr, np.ndarray):
                arr = tuple(list(arr_expr))
            else:
                arr = arr_expr

            if all([isinstance(x, Variable) for x in arr]):
                if not primitive:
                    self.vars = arr
                else:
                    self.vars = [Variable(expr=x()) for x in arr]
            elif all([is_float(x) for x in arr]):
                self.vars = [Variable(expr=x) for x in arr]
            elif len(arr) == 0:
                self.vars = []
            else:
                raise ValueError("list construction either all Vars or floats")
        else:
            expr = arr_expr
            self.vars = []
            
            for i in range(size):
                if is_float(expr):
                    self.vars.append(Variable(expr=expr))
                else:
                    if primitive:
                        val = expr()
                        if not is_float(val):
                            raise ValueError("expression does not evaluate to a float despite primitive specification")
                    else:
                        val = expr

                    self.vars.append(Variable(expr=val))

        self.size = len(self.vars)
        self.grad = ArrayGradient(self)

    def __getitem__(self, idx):
        """mimic list getitem"""

        if isinstance(idx, int):
            return self.vars[idx]
        else:
            return Array(tuple(self.vars[idx]))

    def __setitem__(self, idx, val):
        """mimic list setitem, safe set here"""

        if isinstance(idx, int):
            self.vars[idx].set(val)
        else:
            if isinstance(idx, slice):
                idx = list(range(len(self.vars)))[idx]

            for x, v in zip(idx, val):
                self.vars[x].set(v)

    def set(self, val, primitive=False):
        """
        batch set variables inside the array. rules for
        val and primitive are the same as __init__. uses
        safe setting.

        :param val: value(s) to set the array to
        :param primitive: whether the values should be
            evaluated immediately, making the array
            primitive.
        """
        val = Array(arr_expr=val, primitive=primitive)

        if primitive:
            self[:] = val()
        else:
            self[:] = val

    def clip(self, max_abs):
        """batch clipping for large values"""

        if not all([v._primitive for v in self.vars]):
            raise ValueError("do not clip non primitive vars")
        
        for var in self.vars:
            val = var()
            if abs(val) > max_abs:
                if val < 0:
                    var.set(-1 * max_abs)
                else:
                    var.set(max_abs)

    def __call__(self):
        """call the array to release all values inside"""

        return [x() for x in self.vars]

    @staticmethod
    def _vectorize(self, other, func):
        """
        generic wrapper to vectorize various arithmetic
        functions.

        :param other: float, variable, array, or None,
            used as second number in the operation.
        :param func: the function to vectorize.
        """

        if other == None:
            return Array(tuple([func(x) for x in self.vars]))
        elif is_float(other):
            return Array(tuple([func(x, other) for x in self.vars]))
        elif isinstance(other, Variable):
            return Array(tuple([func(x, other) for x in self.vars]))
        elif isinstance(other, Array) or isinstance(other, list) or isinstance(other, tuple) or isinstance(other, np.ndarray):
            if not isinstance(other, Array):
                other = Array(other)
            
            if self.size != other.size and other.size != 1:
                raise ValueError("cannot operate on arrays of different sizes")
            elif other.size == 1:
                return Array(tuple([func(x, other[0]) for x in self.vars]))
            else:
                return Array(tuple([func(x, y) for x, y in zip(self.vars, other.vars)]))
        else:
            raise ValueError("cannot operate on array to not float/var/array")

    def __add__(self, other):
        return Array._vectorize(self, other, lambda x, y: x + y)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        return Array._vectorize(self, other, lambda x, y: x * y)

    def __truediv__(self, other):
        return Array._vectorize(self, other, lambda x, y: x / y)

    def __pow__(self, other):
        return Array._vectorize(self, other, lambda x, y: x ** y)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return -(self - other)

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return Array._vectorize(self, other, lambda x, y: y / x)

    def __rpow__(self, other):
        return Array._vectorize(self, other, lambda x, y: y ** x)

    def __neg__(self):
        return self * (-1)

    def sum(self):
        """obtain sum"""

        ret = self.vars[0]
        for var in self.vars[1:]:
            ret = ret + var

        return ret

    def mean(self):
        """obtain mean"""

        scale = 1 / self.size
        return self.sum() * scale

def log(var, base=math.e):
    """
    logarithm function. works on variables & arrays

    :param var: the number that is inside the logarithm.
    :param base: optional base, defaults to e=2.71...
    """

    if isinstance(var, Variable):
        if is_float(base):
            base = float(base)
            return Variable(
                expr=lambda: safe_log(var(), base),
                grad=(1 / var) * (1/safe_log(base)) * var.grad
            )
        else:
            raise ValueError("logarithm base must be a float")
    elif isinstance(var, Array):
        return Array._vectorize(var, base, log)
    else:
        return safe_log(var, base)

def exp(var, base=math.e):
    """exp function for completeness"""

    return base ** var

def sin(var):
    """sine function, for variables & arrays"""

    if isinstance(var, Variable):
        return Variable(
            expr=lambda: math.sin(var()),
            grad=var.grad * Variable(expr=lambda: math.cos(var()), _override=True)
        )
    elif isinstance(var, Array):
        return Array._vectorize(var, None, lambda x: sin(x))
    else:
        return math.sin(var)

def cos(var):
    """cosine function, for variables & arrays"""

    if isinstance(var, Variable):
        return Variable(
            expr=lambda: math.cos(var()),
            grad=-var.grad * Variable(expr=lambda: math.sin(var()), _override=True)
        )
    elif isinstance(var, Array):
        return Array._vectorize(var, None, lambda x: cos(x))
    else:
        return math.cos(var)

