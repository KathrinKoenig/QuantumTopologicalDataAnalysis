# Quantum Topological Data Analysis <img width=300 align="right" src="https://user-images.githubusercontent.com/67575757/116423376-eba89780-a840-11eb-843d-00e75ad735bc.jpg">
Implementation of a quantum algorithm to compute topological features of generic (high-dimensional) data clouds in Qiskit.

Topological and geometric features of data are of increasing interest in data analysis to "understand" data and provide complementary global information for modern machine learning pipelines. Therefore, we implement a quantum algorithm to compute such topological features of data clouds, which yields exponential speed-up over all known classical algorithms.

We implement a topological algorithm (https://arxiv.org/abs/1408.3106) on Qiskit to calculate persistent spectra (and from it Betti numbers) of simplicial complexes representing arbitrary (high-dimensional) data. The algorithm builds on the phase estimation algorithm. Our implementation can be run on simulated backends as well as real quantum hardware.

<img width=500 src="https://user-images.githubusercontent.com/67575757/119673446-a4abc300-be3b-11eb-9e98-e7a26e5fc358.png">

## The repository
The implementation of the algorithm can be found in `Code/`. The base file `Code/qtda_module.py` contains basic functions and classes to define the quantum circuit and to analyse data filtrations. The quantum algorithm side is based on the **IBM Qiskit** library (https://github.com/Qiskit), while the library **GUHDI** (https://github.com/GUDHI) is used as a base implementation of (classical) topological data analysis. A **Jupyter-notebook** is provided to explain the basic functionalities of the `QTDA`-implementation and the topological algorithm.

The folder `presentation/` contains a short video presentation of the QTDA implementation, as well as a scientific report describing the main goals and results of our analysis.

## How does TDA work?
As an example TDA is applied to a simple point cloud as shown on the left.
<img width=300 align="left" src="https://user-images.githubusercontent.com/67575757/120495928-6d9a5c00-c3bd-11eb-9dd7-bd660139f6f1.png">

On this point cloud a filtration is done by expanding spheres with radius ε. Points are connected for overlapping spheres. This changes the topology of the data. A simplicial complex is used to represent the topological characteristics of objects. Examples of k-dimensional simplices are: 0 simplex: vertex (point), 1-simplex:edges, 2-simplices are triangles, etc.

<img width=300 align="center" src="https://user-images.githubusercontent.com/67575757/120644603-cf1f0100-c477-11eb-97df-68fe2d9dc6fe.png">

To distinguish topoligical spaces based on the connecitvity of k-dimensional simplicial complexes, Betti numbers are used.
We implemented an algorithm on IBM's quantum computers to find these Betti numbers.

A detailed description can be found in the explanatory notebook.

### Authors

Kathrin König (kathrin.koenig@iaf.fraunhofer.de), Andreas Woitzik (andreas.woitzik@physik.uni-freiburg.de), Eric Brunner (nephts) (eric.brunner@physik.uni-freiburg.de)
