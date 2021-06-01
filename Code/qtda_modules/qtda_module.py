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
from scipy.linalg import expm
from qiskit.extensions import UnitaryGate
from qiskit.circuit.library import PhaseEstimation


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

# # two connected components -------------------

# from scipy import linalg


# S0 = [(0,0,1),(0,1,0),(1,0,0)]
# S1 = [(0,1,1)]
# # S1 = [(0,1,1), (1,1,0),(1,0,1)]
# S2 = []
# # S2 = [(1,1,1)]
# S3 = []

# state_dict = {0: S0, 1: S1, 2: S2, 3: S3, 4: []}

# mat = projected_combinatorial_laplacian(3, 0, state_dict).toarray()
# print(mat)


# U = np.array(linalg.eig(mat)[1])

# print(np.diagonal(U.T @ mat @ U))    

# #%%

# # 2 - simplex -------------------

# from scipy import linalg


# S0 = [(0,0,1),(0,1,0),(1,0,0)]
# S1 = [(0,1,1), (1,1,0), (1,0,1)]
# # S2 = []
# S2 = [(1,1,1)]
# S3 = []

# state_dict = {0: S0, 1: S1, 2: S2, 3: S3, 4: []}

# mat = projected_combinatorial_laplacian(3, 1, state_dict).toarray()
# # print(mat)


# U = np.array(linalg.eig(mat)[1])

# print(np.diagonal(U.T @ mat @ U))

# #%%

# # 3- simplex -------------------

# S0 = [(0,0,0,1),(0,0,1,0),(0,1,0,0),(1,0,0,0)]
# S1 = [(0,0,1,1),(0,1,1,0),(1,1,0,0),(1,0,0,1),(1,0,1,0),(0,1,0,1)]
# S2 = [(0,1,1,1),(1,0,1,1),(1,1,0,1),(1,1,1,0)]
# # S3 = [(1,1,1,1)]
# S3 = []
# S4 = []

# state_dict = {0: S0, 1: S1, 2: S2, 3: S3, 4: S4}

# mat = projected_combinatorial_laplacian(4, 1, state_dict).toarray()
# print(mat)


# U = np.array(linalg.eig(mat)[1])

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
        
        sub_inst = circuit.to_instruction()
        # sub_inst = circuit.to_gate()
        if circuit_name != None:
            sub_inst.name = circuit_name
        qc.append(sub_inst, rest + init)
        return qc


import gudhi as gd  

def simplices_to_states(test_list, n_vertices):
    n_states = len(test_list)
    arr = np.zeros((n_states, n_vertices))
    for k in range(n_states):
        arr[k,test_list[k]] = 1
    return arr

class data_filtration:
    def __init__(self, data, max_dimension, max_edge_length):
        self.skeleton = gd.RipsComplex(points = data, max_edge_length = max_edge_length)
        self.Rips_simplex_tree_sample = self.skeleton.create_simplex_tree(max_dimension = max_dimension)
        self.num_vertices = self.Rips_simplex_tree_sample.num_vertices()
        self.num_simplices = self.Rips_simplex_tree_sample.num_simplices()
        self.filtration = list(self.Rips_simplex_tree_sample.get_filtration())
    
    def get_filtration_states(self, epsilons=None):
        if epsilons == None:
            epsilons = set([x[1] for x in self.filtration])
        filt_dict = {}
        for eps in epsilons:
            helper_list = [x[0] for x in self.filtration if x[1] <= eps]
            filt_dict[eps] = simplices_to_states(helper_list, 4)
        return filt_dict

class QTDA_algorithm(QuantumCircuit):
    def __init__(self, num_eval_qubits, top_order, state_dict, name='QTDA'):

        n_vertices = len(state_dict[0][0])
        
        qr_eval = QuantumRegister(num_eval_qubits, 'eval')
        qr_state = QuantumRegister(n_vertices, 'state')
        qr_copy =QuantumRegister(n_vertices, 'copy')
        super().__init__(qr_eval, qr_state, qr_copy, name=name)
        
        unitary = expm(
            1j*projected_combinatorial_laplacian(n_vertices, top_order, state_dict).toarray()
            )
        
        gate = UnitaryGate(unitary)
        qpe = PhaseEstimation(num_eval_qubits, unitary=gate, iqft=None, name='QPE')

        # qc = initialize_projector(
        #     state_dict[top_order],
        #     circuit=qpe,
        #     initialization_qubits=list(range(num_eval_qubits, num_eval_qubits + n_vertices)),
        #     circuit_name='        QPE        '
        #     )
        
        circuit = qpe
        state = state_dict[top_order]
        initialization_qubits=list(range(num_eval_qubits, num_eval_qubits + n_vertices))
        circuit_name='        QPE        '
        
        n_vertices = len(state[0])
        
        # qr1 = QuantumRegister(n_vertices, name="state_reg")
        # copy_reg = QuantumRegister(n_vertices, name="copy_reg")
        # # clr = ClassicalRegister(circuit.num_clbits, name="cr")
        # if circuit.num_qubits - n_vertices > 0:
        #     anr = QuantumRegister(circuit.num_qubits - n_vertices, name="an_reg")
        #     qc = QuantumCircuit(anr, qr1, copy_reg) #, clr)
        # else:
        #     qc = QuantumCircuit(qr1, copy_reg) #, clr)
        
        state_vec = state_to_vec(state)
        state_vec = state_vec/np.linalg.norm(state_vec)
        self.initialize(state_vec, qr_state)
        
        self.barrier()
        for k in range(n_vertices):
            self.cx(qr_state[k], qr_copy[k])
        self.barrier()
        
        if initialization_qubits == None:
            init = list(range(n_vertices))
        else:
            init = initialization_qubits
        rest = list(set(range(circuit.num_qubits)) - set(init))
        
        sub_inst = circuit.to_instruction()
        # sub_inst = circuit.to_gate()
        if circuit_name != None:
            sub_inst.name = circuit_name
        self.append(sub_inst, rest + init)
            
        # self.helper = qc
        self.eval_qubits = list(range(num_eval_qubits))
        # self.append(qc, self.qubits)

def Q_persistent_top_spectra(data, max_dimension, max_edge_length, num_eval_qubits, epsilons=None):
        da_fil = data_filtration(data, max_dimension, max_edge_length)
        filt_dict = da_fil.get_filtration_states(epsilons=epsilons)
        
        state_dict = {}
        for key in filt_dict.keys():
            state_dict[key] = {}
            for k in range(1,max_dimension+1):
                mask = np.sum(filt_dict[key],axis=1) == k
                state_dict[key][k] = filt_dict[key][mask, :]
        
        # self.q_circuit = QTDA_algorithm(num_eval_qubits, top_order, state_dict)
        return state_dict
        
        
# #%%

# test_sample = np.array([
#     [0.,0.],
#     [1.,0.],
#     [1.,1.],
#     [1.,0.]
#     ])

# df = data_filtration(test_sample, max_dimension=4, max_edge_length=2)

# # a = df.filtration
# # print(list(a))
# aa = (df.get_filtration_states([0.1, 1.1, 1.5]))
# aaa = (df.get_filtration_states())

# b = Q_persistent_top_spectra(test_sample, max_dimension=4, max_edge_length=2, num_eval_qubits=2, epsilons=[0.1, 1.1, 1.5])

# #%%
# print(b[0.1])

# print(aa[0.1])
# print(aaa[0])
# #%%      
        
# #%%

# n_vertices = 3
# S0 = [(0,0,1),(0,1,0), (1,0,0)]
# S_test = [(0,1,0,0),(0,0,1,1)]
# S1 = [(0,1,1),(1,1,0),(1,0,1)]
# S2 = []
# #S2 = [(1,1,1)]
# S3 = []

# state_dict = {0: S0, 1: S1, 2: S2, 3: S3}

# n_vertices = 3
# S0 = [(0,0,1),(0,1,0), (1,0,0)]
# S_test = [(0,1,0,0),(0,0,1,1)]
# S1 = [(0,1,1),(1,1,0),(1,0,1)]
# S2 = []
# #S2 = [(1,1,1)]
# S3 = []

# state_dict = {0: S0, 1: S1, 2: S2, 3: S3}

# qc = QTDA_algorithm(5, 1, state_dict)

# #%%

# a = qc.helper
# print(a.qubits)
# print()
# print(a._qubits)

# #%%






