# Code

### Dependencies:
numpy  
gudhi  
itertools  
scipy  
math  
qiskit  

The basic Python module `qtda_module.py` contains the base classes generating the quantum circuit of the QTDA-algorithm as well as the data filtration of point data. It builds on the libraries **IBM Qiskit** library (https://github.com/Qiskit) for quantum computing and **GUHDI** (https://github.com/GUDHI) for classical TDA.

The [Introductory Jupyter-Notebook](https://github.com/KathrinKoenig/QuantumTopologicalDataAnalysis/blob/main/Code/Introductory_notebook.ipynb) explains basic functionalities of the algorithm and the filtration of simplicial complexes describing topological and geometric features of the data points on different scales.

To analyse topological features of data clouds on different scales, we have to combine this with the data filtration procedure, explained in the beginning of the introductory notebook. This is done in the [Quantum persistent topological spectra Notebook](https://github.com/KathrinKoenig/QuantumTopologicalDataAnalysis/blob/main/Code/quantum_persistent_top_spectra.ipynb).
