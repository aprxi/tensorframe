
MODULE = tensorframe

pull:
	git pull && \
	git submodule update --recursive

build_cpp:
	make -C ./cpp libio && \
	make -C ./cpp libbinops && \
	cp cpp/build/*.so tensorframe/core/libs/ 

build_python:
	python3 setup.py sdist bdist_wheel


lint:
	pylint --rcfile=.pylintrc $(MODULE) -f parseable 
	# to be enabled later (requiring another round of cleaning up)
	#&& flake8 $(MODULE) \
	#&& pydocstyle $(MODULE) \
	#&& mypy -m $(MODULE)

test: lint

build: build_cpp test build_python

clean: 
	rm -rf ./test/__pycache__ ./tensorframe/__pycache__ ./tensorframe/core/__pycache__ ./tensorframe/core/libs/__pycache__
	rm -rf ./build ./dist ./tensorframe.egg-info
	rm -rf ./tensorframe/core/libs/*pyc ./tensorframe/core/libs/*so
	make -C ./cpp clean
