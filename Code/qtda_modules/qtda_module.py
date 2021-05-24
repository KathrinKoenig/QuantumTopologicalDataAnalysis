#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 11:04:02 2021

@author: ericbrunner
"""

import numpy as np
import itertools as it
from scipy.sparse import csr_matrix
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, AncillaRegister

def vec_to_state(vec):
    n_vertices = int(np.log2(len(vec)))
    basis_list = list(it.product(range(2), repeat=n_vertices))
    indices = np.nonzero(vec)[0]
    return {basis_list[i]:vec[i] for i in indices}

def state_to_vec(state):
    ''' state is a list of state '''
    n_vertices = len(state[0])
    basis_list = list(it.product(range(2), repeat=n_vertices))
    basis_dict = {basis_list[i]: i for i in range(2**n_vertices)}
    vec = np.zeros(2**n_vertices)
    for s in state:
        vec[basis_dict[s]] = 1
    return vec

def boundary_operator(x_tuple):
    x_np = np.array(x_tuple)
    indices = np.nonzero(x_np)[0]
    dictionary = {}
    k = len(indices)
    for i in range(k):
        helper = x_np.copy()
        helper[indices[k-1-i]] = 0
        dictionary[tuple(helper)] = (-1)**(i)
    return dictionary

def boundary_operator_dict(n_vertices):
    dictionary = {}
    dictionary[(tuple([0 for i in range(n_vertices)]),tuple([0 for i in range(n_vertices)]))] = 0
    for b in it.product(range(2), repeat=n_vertices):
        helper = boundary_operator(b)
        for key in helper.keys():
            dictionary[(tuple(b),key)] = helper[key]
    return dictionary

def boundary_operator_dict_k(n_vertices, k):
    dictionary = {}
    dictionary[(tuple([0 for i in range(n_vertices)]),tuple([0 for i in range(n_vertices)]))] = 0
    for b in it.product(range(2), repeat=n_vertices):
        if np.sum(b) == k:
            helper = boundary_operator(b)
            for key in helper.keys():
                dictionary[(tuple(b),key)] = helper[key]
    return dictionary

def boundary_operator_crsmat(n_vertices):
    dictionary = boundary_operator_dict(n_vertices)
    basis_list = list(it.product(range(2), repeat=n_vertices))
    basis_dict = {basis_list[i]: i for i in range(2**n_vertices)}
    col = np.array([basis_dict[index[0]] for index in dictionary.keys()])
    row = np.array([basis_dict[index[1]] for index in dictionary.keys()])
    data = np.array(list(dictionary.values()))
    return csr_matrix((data, (row, col)), shape=(2**n_vertices, 2**n_vertices))

def boundary_operator_crsmat_k(n_vertices, k):
    dictionary = boundary_operator_dict_k(n_vertices, k)
    basis_list = list(it.product(range(2), repeat=n_vertices))
    basis_dict = {basis_list[i]: i for i in range(2**n_vertices)}
    col = np.array([basis_dict[index[0]] for index in dictionary.keys()])
    row = np.array([basis_dict[index[1]] for index in dictionary.keys()])
    data = np.array(list(dictionary.values()))
    return csr_matrix((data, (row, col)), shape=(2**n_vertices, 2**n_vertices))

def combinatorial_laplacian(n_vertices, k):
    delta_k = boundary_operator_crsmat_k(n_vertices, k)
    delta_kplus1 = boundary_operator_crsmat_k(n_vertices, k+1)
    return delta_k.conj().T @ delta_k + delta_kplus1 @ delta_kplus1.conj().T

def initialize_projector(state, circuit=None):
    '''initializes projector onto subspace spanned by list of states'''
    '''input circuit has to have classical register'''
    
    n_vertices = len(state[0])
    qr1 = QuantumRegister(n_vertices, name="qr1")
    anr = AncillaRegister(n_vertices, name="ancilla")
    qc = QuantumCircuit(qr1,anr)
    
    state_vec = state_to_vec(state)
    state_vec = state_vec/np.linalg.norm(state_vec)
    qc.initialize(state_vec, qr1)
    qc.barrier()
    for k in range(n_vertices):
        qc.cx(qr1[k],anr[k])
    qc.barrier()
    if circuit == None:
        return qc
    else:
        qc.add_register(ClassicalRegister(circuit.num_clbits, name="cr"))
        return qc.compose(circuit, qubits=qr1)









