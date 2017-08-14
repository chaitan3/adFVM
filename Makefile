all: buildall
buildall:
	python2 setup.py build
	cp build/lib*/adFVM/compat/cfuncs.so adFVM/compat
	cd adFVM/cpp && make
install: buildall
	python2 setup.py install --prefix=~/.local
clean:
	python2 setup.py clean
	cd adFVM/cpp && make clean
