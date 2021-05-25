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
        helper[indices[k-1-i]] = 0.
        dictionary[tuple(helper)] = (-1.)**(i)
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
        if np.sum(b) == k+1:
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

def projector_onto_state(n_vertices, state):
    basis_list = list(it.product(range(2), repeat=n_vertices))
    basis_dict = {basis_list[i]: i for i in range(2**n_vertices)}
    indices = [basis_dict[s] for s in state]
    return csr_matrix((np.ones(len(indices)), (indices, indices)), shape=(2**n_vertices, 2**n_vertices))
 
def projected_combinatorial_laplacian(n_vertices, k, state_dict):
    P_k = projector_onto_state(n_vertices, state_dict[k])
    P_kp1 = projector_onto_state(n_vertices, state_dict[k+1])
    delta_k = boundary_operator_crsmat_k(n_vertices, k) @ P_k
    delta_kplus1 = boundary_operator_crsmat_k(n_vertices, k+1) @ P_kp1
    return delta_k.conj().T @ delta_k + delta_kplus1 @ delta_kplus1.conj().T 
  
# #%%

# print(combinatorial_laplacian(3, 1).toarray())

# state_k = [(0,1,1), (1,1,0), (1,0,1)]
# state_kp1 = [(1,1,1)]

# state_dict = {1: [(0,1,1), (1,1,0), (1,0,1)], 2: [(1,1,1)]}

# mat = projected_combinatorial_laplacian(3, 1, state_dict).toarray()
# print(mat)

# #%%
# from scipy import linalg
# U = np.array(linalg.eig(mat)[1])

# print(U)

# print(np.diagonal(U.T @ mat @ U))

# #%%


def initialize_projector(state, circuit=None, initialization_qubits=None, circuit_name=None):
    '''initializes projector onto subspace spanned by list of states'''
    '''input circuit has to have classical register'''
    
    if circuit == None:
        n_vertices = len(state[0])
        qr1 = QuantumRegister(n_vertices, name="state_reg")
        copy_reg = QuantumRegister(n_vertices, name="copy_reg")
        qc = QuantumCircuit(qr1,copy_reg)
        
        state_vec = state_to_vec(state)
        state_vec = state_vec/np.linalg.norm(state_vec)
        qc.initialize(state_vec, qr1)
        qc.barrier()
        for k in range(n_vertices):
            qc.cx(qr1[k],copy_reg[k])
        qc.barrier()
        return qc
    else:
        n_vertices = len(state[0])
        qr1 = QuantumRegister(n_vertices, name="state_reg")
        copy_reg = QuantumRegister(n_vertices, name="copy_reg")
        # clr = ClassicalRegister(circuit.num_clbits, name="cr")
        if circuit.num_qubits - n_vertices > 0:
            anr = QuantumRegister(circuit.num_qubits - n_vertices, name="an_reg")
            qc = QuantumCircuit(anr, qr1, copy_reg) #, clr)
        else:
            qc = QuantumCircuit(qr1, copy_reg) #, clr)
        
        state_vec = state_to_vec(state)
        state_vec = state_vec/np.linalg.norm(state_vec)
        qc.initialize(state_vec, qr1)
        qc.barrier()
        for k in range(n_vertices):
            qc.cx(qr1[k],copy_reg[k])
        qc.barrier()
        
        if initialization_qubits == None:
            init = list(range(n_vertices))
        else:
            init = initialization_qubits
        rest = list(set(range(circuit.num_qubits)) - set(init))
        
        # sub_inst = circuit.to_instruction()
        sub_inst = circuit.to_gate()
        if circuit_name != None:
            sub_inst.name = circuit_name
        qc.append(sub_inst, rest + init)
        return qc
        # return qc.compose(circuit, qubits=qr1)


# #%%

# qr_test = QuantumRegister(5, name="qr_test3")
# cr_test = ClassicalRegister(4, name="cr_test3")
# qc_test = QuantumCircuit(qr_test) #,cr_test)
# qc_test.barrier()
# qc_test.h(0)
# qc_test.h(1)
# qc_test.h(2)
# qc_test.barrier()
# # qc_test.measure(qr_test[2],cr_test[3])
# qc_test.draw('mpl')


# #%%

# S1 = [(0,0,1,1),(0,1,1,0), (1,1,0,0), (1,0,0,1), (1,0,1,0)]

# qc = initialize_projector(S1, circuit=qc_test, initialization_qubits=None, circuit_name='test_circuit')
# qc.add_register(ClassicalRegister(4, name="cr_test3"))
# qc.measure(qc.qubits[0],3)
# qc.draw('mpl')

# #%%

# print(qc.qregs)

# #%%
# import qiskit

# aa = qiskit.circuit.Qubit(QuantumRegister(1, 'anr'),0)
# #%%

# a = {1,4,3}
# b = {5,6,2}
# print(a.union(b) )
