setup:
	pip install -r requirements.txt
	cython pyhacrf/*.pyx
	python setup.py install

clean:
	rm -rf __pycache__
