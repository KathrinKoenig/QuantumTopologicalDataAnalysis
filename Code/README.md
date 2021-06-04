# Code

### Dependencies:
- [NumPy](https://numpy.org/)  
- [GUDHI](https://gudhi.inria.fr/)  
- [itertools](https://docs.python.org/3/library/itertools.html)  
- [SciPy](https://www.scipy.org/)  
- [Qiskit](https://qiskit.org/)  

### Notebooks:
The basic Python module `qtda_module.py` contains the base classes generating the quantum circuit of the QTDA-algorithm as well as the data filtration of point data. It builds on the libraries **IBM Qiskit** library (https://github.com/Qiskit) for quantum computing and **GUHDI** (https://github.com/GUDHI) for classical topological data analysis.

The `Introductory_notebook.ipynb` explains basic functionalities of the algorithm and the filtration of simplicial complexes describing topological and geometric features of the data points on different scales.

To analyse topological features of data clouds on different scales, we have to combine this with the data filtration procedure, explained in the beginning of the introductory notebook. This is done in the notebook `quantum_persistent_top_spectra.ipynb`

The third notebook `top_spectra-real_q-device.ipynb` contains the application of the algorithm on a real IBM quantum devise for a small example simplicial complex.


[Quantum persistent topological spectra Notebook](https://github.com/KathrinKoenig/QuantumTopologicalDataAnalysis/blob/main/Code/quantum_persistent_top_spectra.ipynb).



[Introductory Jupyter-Notebook](https://github.com/KathrinKoenig/QuantumTopologicalDataAnalysis/blob/main/Code/Introductory_notebook.ipynb) 


