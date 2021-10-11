import math

def is_float(inn):
    try:
        float(inn)
        return True
    except:
        return False

class _CallableArray():
    def __init__(self, arr):
        self.arr = arr
    
    def __getitem__(self, idx):
        return self.arr.__getitem__(idx)
    
    def __call__(self):
        return [x() for x in self.arr]

class Gradient():
    def __init__(self, vars, exprs):
        self.vars = vars
        self.exprs = exprs

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            ret = []
            for x in idx:
                ret.append(self[x])
            return _CallableArray(ret)
        elif isinstance(idx, Variable):
            if idx._primitive:
                return self.exprs[self.vars.index(idx)]
            else:
                raise NotImplementedError("gradient indexing only implemented for primitives")
        else:
            raise ValueError("can only index gradient by list of Variables or single Variable")

    def __add__(self, other):
        if isinstance(other, Variable):
            other = other.grad
            combined_vars = list(set(self.vars + other.vars))
            combined_exprs = []
            for var in combined_vars:
                if var in self.vars and var in other.vars:
                    combined_exprs.append(lambda: self[var] + other[var])
                elif var in self.vars and var not in other.vars:
                    combined_exprs.append(self[var])
                elif var not in self.vars and var in other.vars:
                    combined_exprs.append(other[var])
            return Gradient(combined_vars, combined_exprs)
        else:
            raise ValueError("gradients can only be added to other gradients")

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if is_float(other):
            other = float(other)
            return Gradient(vars, [lambda: expr() * other for expr in exprs])
        else:
            raise ValueError("gradients can only be multiplied with floats")
    
    def __truediv__(self, other):
        if is_float(other):
            other = float(other)
            return Gradient(vars, [lambda: expr() / other for expr in exprs])
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

    def __call__(self):
        return self

class Variable():
    def __init__(self, expr=None, grad=None):
        is_primitive_def = is_float(expr) or (expr == None)

        if is_primitive_def and grad != None:
            return ValueError("primitive Variables cannot have gradients")
        if (not is_primitive_def) and grad == None:
            return ValueError("non-primitive Variables must have gradients")

        self._primitive = is_primitive_def

        def not_set_yet():
            raise ValueError("this primitive Variable has not been set yet")

        if self._primitive:
            if expr == None:
                self.eval = lambda: not_set_yet()
            else:
                self.eval = lambda: expr
            self.grad = Gradient(vars=[self], exprs=[lambda: 1])
        else:
            self.eval = expr
            self.grad = grad

    def set(self, val):
        # safe reassignments
        if is_float(val):
            if not self._primitive:
                raise ValueError("only primitive Variables can be assigned to floats")
            self.eval = lambda: float(val)
        elif isinstance(val, Variable):
            def cyclical_assignment():
                raise ValueError("cyclical assignment")

            self.expr = lambda: cyclical_assignment()
            val()

            if self._primitive:
                self._primitive = False

            self.eval = val.eval
            self.grad = val.grad
        else:
            raise ValueError("cannot set Variables to anything other than float or Variable")

    def __call__(self):
        return self.eval()

    @staticmethod
    def set_vars(vars, vals):
        for (var, val) in zip(vars, vals):
            var.set(val)
        return

    def __add__(self, other):
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
        if is_float(other):
            other = float(other)
            return Variable(
                expr=lambda: self() * other,
                grad=self.grad * other
            )
        elif isinstance(other, Variable):
            return Variable(
                expr=lambda: self() * other(),
                grad=self.grad * other() + other.grad * self()
            )
        else:
            raise ValueError("Variables can only be multiplied with floats or other Variables")

    def __truediv__(self, other):
        if is_float(other):
            other = float(other)
            return Variable(
                expr=lambda: self() / other,
                grad=self.grad / other
            )
        elif isinstance(other, Variable):
            return self * (1 / other)
        else:
            raise ValueError("Variables can only be divided by floats or other Variables")

    def __pow__(self, other):
        if is_float(other):
            other = float(other)
            return Variable(
                expr=lambda: self() ** other,
                grad=(self() ** (other - 1)) * self.grad
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
                expr=lambda: other / self(),
                grad=(-1 / (self())**2) * self.grad * other
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
                grad=(other ** self()) * math.log(other) * self.grad
            )
        else:
            raise ValueError("Variables can only be the exponent of a float")

    def __neg__(self):
        return self * (-1)

    @staticmethod
    def log(var, base=math.e):
        if is_float(base):
            base = float(base)
            return Variable(
                expr=lambda: math.log(var(), base),
                grad=(1 / var()) * var.grad
            )
        else:
            raise ValueError("logarithm base must be a float")

    @staticmethod
    def pow(var, base=math.e):
        return base ** var

    @staticmethod
    def sin(var):
        return Variable(
            expr=lambda: math.sin(var()),
            grad=math.cos(var()) * var.grad
        )

    @staticmethod
    def cos(var):
        return Variable(
            expr=lambda: math.cos(var()),
            grad=-math.sin(var()) * var.grad
        )
