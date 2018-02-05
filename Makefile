PYTHON=python
all: buildall
buildall:
	$(PYTHON) setup.py build
	cp build/lib*/adFVM/compat/cfuncs*.so adFVM/compat
	cp build/lib*/adFVM/cpp/*.so adFVM/cpp
install: buildall
	$(PYTHON) setup.py install --prefix=~/.local
clean:
	rm -rf adFVM/compat/*.so adFVM/cpp/*.so
	$(PYTHON) setup.py clean
