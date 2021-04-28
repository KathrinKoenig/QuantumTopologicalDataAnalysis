# Quantum Topological Data Analysis <img width=100 align="right" src="https://user-images.githubusercontent.com/67575757/116423376-eba89780-a840-11eb-843d-00e75ad735bc.jpg">
Implementation of a quantum algorithm to compute topological features of generic (high-dimensional) data clouds in Qiskit.

Topological and geometric features of data are of increasing interest in data analysis to "understand" data and provide complementary global information for modern machine learning pipelines. Therefore, we implement a quantum algorithm to compute such topological features of data clouds, which yields exponential speed-up over all known classical algorithms.

We implement a topological algorithm (https://arxiv.org/abs/1408.3106) on Qiskit to calculate persistent spectra (and from it Betti numbers) of simplicial complexes representing arbitrary (high-dimensional) data. The algorithm builds on the phase estimation algorithm. After successful implementation we apply the algorithm to real datasets on IBM quantum devises. 

<img width=500 src="https://user-images.githubusercontent.com/67575757/116424254-a769c700-a841-11eb-8737-b1b1ff9a4053.jpg">
