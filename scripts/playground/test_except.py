def foo():
    1/0

#foo()

#try:
#    foo()
#except:
#    raise

try:
    foo()
except Exception as e:
    a = (2,2)
    e.args += (a,)
    raise


