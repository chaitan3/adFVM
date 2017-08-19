all: buildall
buildall:
	python2 setup.py build
	cp build/lib*/adFVM/compat/cfuncs.so adFVM/compat
	cp build/lib*/adFVM/cpp/*.so adFVM/cpp
install: buildall
	python2 setup.py install --prefix=~/.local
clean:
	rm -rf adFVM/compat/*.so adFVM/cpp/*.so
	python2 setup.py clean
