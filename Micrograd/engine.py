import math

class Value:
    def __init__(self,data, _children=(), op='', label=''):
        self.data=data
        self.grad=0.0

        self._prev=set(_children)
        self._op=op
        self.label=label
        self._backward=lambda: None

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _children=(self, other), op='+', label=f'({self.label}+{other.label})')

        def backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other), op='*', label=f'({self.label}*{other.label})')

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)

    def __pow__(self, exponent):
        out = Value(self.data ** exponent, _children=(self,), op=f'**{exponent}')

        def backward():
            self.grad += exponent * self.data ** (exponent - 1) * out.grad
        out._backward = backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, _children=(self,), op='ReLU', label=f'ReLU({self.label})')
        def backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = backward
        return out

    def tanh(self):
        t = (math.exp(2*self.data) - 1) / (math.exp(2*self.data) + 1)
        out = Value(t, _children=(self,), op='tanh', label=f'tanh({self.label})')
        def backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = backward
        return out

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()