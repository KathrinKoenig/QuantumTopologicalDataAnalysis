#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 11:04:02 2021

@author: ericbrunner
"""

import numpy as np
import gudhi as gd
import itertools as it
from scipy.sparse import csr_matrix
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from scipy.linalg import expm
from qiskit.circuit.library import PhaseEstimation
from qiskit.extensions import UnitaryGate
from qiskit import Aer
from qiskit import execute


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
    return csr_matrix(
        (np.ones(len(indices)), (indices, indices)),
        shape=(2**n_vertices, 2**n_vertices)
        )

def projected_combinatorial_laplacian(n_vertices, k, state_dict):
    P_k = projector_onto_state(n_vertices, state_dict[k])
    P_kp1 = projector_onto_state(n_vertices, state_dict[k+1])
    delta_k = boundary_operator_crsmat_k(n_vertices, k) @ P_k
    delta_kplus1 = boundary_operator_crsmat_k(n_vertices, k+1) @ P_kp1
    return delta_k.conj().T @ delta_k + delta_kplus1 @ delta_kplus1.conj().T



def initialize_projector(state, circuit=None, initialization_qubits=None, circuit_name=None):
    '''
    initializes projector onto subspace spanned by list of states
    input circuit has to have classical register
    '''
    
    if circuit == None:
        n_vertices = len(state[0])
        qr1 = QuantumRegister(n_vertices, name="state_reg")
        copy_reg = QuantumRegister(n_vertices, name="copy_reg")
        qc = QuantumCircuit(qr1,copy_reg)
        
        state_vec = state_to_vec(state)
        state_vec = state_vec/np.linalg.norm(state_vec)
        qc.initialize(state_vec, qr1)
        # qc.barrier()
        for k in range(n_vertices):
            qc.cx(qr1[k],copy_reg[k])
        # qc.barrier()
    else:
        n_vertices = len(state[0])
        qr1 = QuantumRegister(n_vertices, name="state_reg")
        copy_reg = QuantumRegister(n_vertices, name="copy_reg")
        if circuit.num_qubits - n_vertices > 0:
            anr = QuantumRegister(circuit.num_qubits - n_vertices, name="an_reg")
            qc = QuantumCircuit(anr, qr1, copy_reg) #, clr)
        else:
            qc = QuantumCircuit(qr1, copy_reg) #, clr)
        
        state_vec = state_to_vec(state)
        state_vec = state_vec/np.linalg.norm(state_vec)
        qc.initialize(state_vec, qr1)
        # qc.barrier()
        for k in range(n_vertices):
            qc.cx(qr1[k],copy_reg[k])
        # qc.barrier()
        
        if initialization_qubits == None:
            init = list(range(n_vertices))
        else:
            init = initialization_qubits
        rest = list(set(range(circuit.num_qubits)) - set(init))
        
        sub_inst = circuit.to_instruction()
        if circuit_name != None:
            sub_inst.name = circuit_name
        qc.append(sub_inst, rest + init)
        
    return qc




def simplices_to_states(test_list, n_vertices):
    n_states = len(test_list)
    arr = np.zeros((n_states, n_vertices))
    for k in range(n_states):
        arr[k,test_list[k]] = 1
    return arr

class data_filtration:
    def __init__(self, data=None, distance_matrix=None, max_dimension=None, max_edge_length=None):
        '''
        data: point data given by (number_data_points x dim_points)-numpy-array
        distance_matrix: (n x n)-numpy-array describing the pair-wise distances of n data points
        
        Either one of them has to be given!
        '''
        
        if data is not None:
            self.skeleton = gd.RipsComplex(
                points = data, 
                max_edge_length = max_edge_length
                )
        elif distance_matrix is not None:
            self.skeleton = gd.RipsComplex(
                distance_matrix = distance_matrix, 
                max_edge_length = max_edge_length
                )
        else:
            print('Either point data or distance matrix has to be provided!')
            
        self.Rips_simplex_tree = self.skeleton.create_simplex_tree(max_dimension = max_dimension)
        self.num_vertices = self.Rips_simplex_tree.num_vertices()
        self.num_simplices = self.Rips_simplex_tree.num_simplices()
        self.filtration = list(self.Rips_simplex_tree.get_filtration())
    
    def get_filtration_states(self, epsilons=None):
        if epsilons == None:
            epsilons = set([x[1] for x in self.filtration])
        filt_dict = {}
        for eps in epsilons:
            helper_list = [x[0] for x in self.filtration if x[1] <= eps]
            filt_dict[eps] = simplices_to_states(helper_list, self.num_vertices)
        return filt_dict
    
    def plot_persistence_diagram(self):
        BarCodes = self.Rips_simplex_tree.persistence()
        gd.plot_persistence_diagram(BarCodes, legend=True)






def controledU(U, qc, num_eval_qubits, n_vertices):#, k, state_dict):
    unit = U
#     unit = scipy.linalg.expm(1j*qtda.projected_combinatorial_laplacian(n_vertices, k, state_dict).toarray())
    gate = UnitaryGate(unit)
    for counting_qubit in range(num_eval_qubits):
        qc.append(gate.control(1), [counting_qubit] + list(range(num_eval_qubits,num_eval_qubits+n_vertices)))
        unit = unit@unit
        gate = UnitaryGate(unit)
    return qc

def qft_dagger(qc, n):
    """n-qubit QFTdagger the first n qubits in circ"""
    # Don't forget the Swaps!
    for qubit in range(n//2):
        qc.swap(qubit, n-qubit-1)
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi/float(2**(j-m)), m, j)
        qc.h(j)
        
def qpe_total(num_eval_qubits, n_vertices, unitary): #, k, state_dict):
    qc = QuantumCircuit(num_eval_qubits + n_vertices)
    for qubit in range(num_eval_qubits):
        qc.h(qubit)
    controledU(unitary, qc, num_eval_qubits, n_vertices) #, k, state_dict)
    # Apply inverse QFT
    qft_dagger(qc, num_eval_qubits)
    return qc


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
        
        '''
        we can also directly use the PhaseEstimation routine from Qiskit, which, however,
        leads to a larger circuit depth than our designed qpe_total implementation 
        
        '''
        
        # gate = UnitaryGate(unitary)
        # qpe = PhaseEstimation(num_eval_qubits, unitary=gate, iqft=None, name='QPE')
        qpe = qpe_total(num_eval_qubits, n_vertices, unitary)
        
        sub_inst = qpe.to_instruction()
        sub_inst.name = '        QPE        '
        
        state_vec = state_to_vec(state_dict[top_order])
        state_vec = state_vec/np.linalg.norm(state_vec)
        self.initialize(state_vec, qr_state)
        
        # self.barrier()
        for k in range(n_vertices):
            self.cx(qr_state[k], qr_copy[k])
        # self.barrier()
        
        self.append(sub_inst, list(range(num_eval_qubits + n_vertices)))
        self.eval_qubits = list(range(num_eval_qubits))

class Q_top_spectra:
    def __init__(self, state_dict, num_eval_qubits=6, shots=1000):
        '''
        data: point data given by (number_data_points x dim_points)-numpy-array
        distance_matrix: (n x n)-numpy-array describing the pair-wise distances of n data points
        
        Either one of them has to be given!
        '''
        self.shots = shots
        self.state_dict = state_dict
        
        self.counts = {}
        for top_order in self.state_dict.keys():
            print('Topological order: ', top_order)
            if (len(self.state_dict[top_order])==0):
                print("calculation terminated because no simplex of dimension %s" %(top_order))
                break
            qc = QTDA_algorithm(num_eval_qubits, top_order, self.state_dict)
            qc.add_register(ClassicalRegister(num_eval_qubits, name="phase"))
            for q in qc.eval_qubits:
                qc.measure(q,q)
            
            backend = Aer.get_backend('qasm_simulator')
            job = execute(qc, backend, shots=self.shots)
            self.counts[top_order] = job.result().get_counts(qc)
    
    def get_counts(self):
        return self.counts
    
    def get_spectra(self, chop=None):
        '''
        defaul chop: eigenvalues with less then 2% of all shots are regarded as noise (can be adjusted)
        exception: the eigenspace of eigenvalue 0 is important, even if it is 0-dimensional; hence, this is always included
        '''
        if chop == None:
            chop = self.shots/50   
        eigenvalue_dict = {}
        for top_order in self.counts.keys():
            eigenvalue_dict[top_order] = {}
            eigenvalue_dict[top_order][0.0] = 0
            vals = np.fromiter(self.counts[top_order].values(), dtype=float)
            keys = list(self.counts[top_order].keys())
            indices = np.where(vals >= chop)
            new_vals = vals[indices]
            new_keys = [keys[i] for i in indices[0]]
            for i in range(len(new_keys)):
                eigenvalue_dict[top_order][2*np.pi*int(new_keys[i], 2)/int(len(new_keys[i])*'1',2)] = (
                    new_vals[i] * len(self.state_dict[top_order])/self.shots
                    )
        return eigenvalue_dict
    
class Q_persistent_top_spectra:
    def __init__(
            self,
            data=None,
            distance_matrix=None,
            max_dimension=None,
            max_edge_length=None,
            num_eval_qubits=6,
            shots=1000,
            epsilons=None
            ):
        '''
        data: point data given by (number_data_points x dim_points)-numpy-array
        distance_matrix: (n x n)-numpy-array describing the pair-wise distances of n data points
        
        Either one of them has to be given!
        '''
        self.shots = shots
        self.filt_dict = data_filtration(
            data=data, 
            distance_matrix=distance_matrix,
            max_dimension=max_dimension, 
            max_edge_length=max_edge_length
            ).get_filtration_states(epsilons=epsilons)
        self.state_dict = {}
        for key in sorted(self.filt_dict.keys()):
            self.state_dict[key] = {}
            for k in range(1,max_dimension+1):
                mask = np.sum(self.filt_dict[key],axis=1) == k
                self.state_dict[key][k-1] = [
                    tuple(s) 
                    for s in self.filt_dict[key][mask, :]
                    ]
            self.state_dict[key][max_dimension] = [] # an empty state has to be included on order max_dimension for consistency 
        
        self.counts = {}
        for eps in self.state_dict.keys():
            print()
            print('Filtration scale: ', eps)
            print()
            self.counts[eps] = {}
            for top_order in self.state_dict[eps].keys():
                print('Topological order: ', top_order)
                if (len(self.state_dict[eps][top_order])==0):
                    print("calculation terminated because no simplex of dimension %s" %(top_order))
                    break
                
                qc = QTDA_algorithm(
                    num_eval_qubits,
                    top_order,
                    self.state_dict[eps]
                    )
                qc.add_register(ClassicalRegister(num_eval_qubits, name="phase"))
                for q in qc.eval_qubits:
                    qc.measure(q,q)
                backend = Aer.get_backend('qasm_simulator')
                job = execute(qc, backend, shots=self.shots)
                self.counts[eps][top_order] = job.result().get_counts(qc)
    
    def get_counts(self):
        return self.counts
    
    def get_eigenvalues(self, chop=None):
        '''
        defaul chop: eigenvalues with less then 2% of all shots are regarded as noise (can be adjusted)
        exception: the eigenspace of eigenvalue 0 is important, even if it is 0-dimensional; hence, this is always included
        '''
        if chop == None:
            chop = self.shots/50
            
        eigenvalue_dict = {}
        for eps in self.counts.keys():
            eigenvalue_dict[eps] = {}
            for top_order in self.counts[eps].keys():
                eigenvalue_dict[eps][top_order] = {}
                eigenvalue_dict[eps][top_order][0.0] = 0
                vals = np.fromiter(self.counts[eps][top_order].values(), dtype=float)
                keys = list(self.counts[eps][top_order].keys())
                indices = np.where(vals >= chop)
                new_vals = vals[indices]
                new_keys = [keys[i] for i in indices[0]]
                for i in range(len(new_keys)):
                    eigenvalue_dict[eps][top_order][2*np.pi*int(new_keys[i], 2)/int(len(new_keys[i])*'1',2)] = (
                        new_vals[i] * len(self.state_dict[eps][top_order])/self.shots
                        )
        
        return eigenvalue_dict
                
    
