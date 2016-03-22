import theano

a = theano.tensor.scalar()
b = theano.tensor.scalar()
c = theano.shared(4)

f = theano.function([a,b], a*b)

print f(2, 3)
print f(2, c.get_value())
print f(2, c)
