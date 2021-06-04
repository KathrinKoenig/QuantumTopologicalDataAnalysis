#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 11:04:02 2021

@author: Eric Brunner, Kathrin König, Andreas Woitzik
"""

import itertools as it
import numpy as np
import gudhi as gd
from scipy.sparse import csr_matrix
from scipy.linalg import expm
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import Aer
from qiskit import execute
from qiskit.extensions import UnitaryGate


def vec_to_state(vec):
    '''converts vec to state'''
    n_vertices = int(np.log2(len(vec)))
    basis_list = list(it.product(range(2), repeat=n_vertices))
    indices = np.nonzero(vec)[0]
    return {basis_list[i]: vec[i] for i in indices}


def state_to_vec(state):
    '''state is a list of state'''
    n_vertices = len(state[0])
    basis_list = list(it.product(range(2), repeat=n_vertices))
    basis_dict = {basis_list[i]: i for i in range(2**n_vertices)}
    vec = np.zeros(2**n_vertices)
    for stat in state:
        vec[basis_dict[stat]] = 1
    return vec


def boundary_operator(x_tuple):
    '''calculates the boundary of a simplex desribed by x_tuple'''
    x_np = np.array(x_tuple)
    indices = np.nonzero(x_np)[0]
    dictionary = {}
    k = len(indices)
    for i in range(k):
        helper = x_np.copy()
        helper[indices[k-1-i]] = 0.0
        dictionary[tuple(helper)] = (-1.)**(i)
    return dictionary


def boundary_operator_dict(n_vertices):
    '''
    returns the boundary operator on n vertices as a dictionary
    '''
    dictionary = {}
    dictionary[
        (
            tuple([0]*n_vertices),
            tuple([0]*n_vertices)
            )
        ] = 0
    for bound in it.product(range(2), repeat=n_vertices):
        helper = boundary_operator(bound)
        for key in helper:
            dictionary[(tuple(bound), key)] = helper[key]
    return dictionary


def boundary_operator_dict_k(n_vertices, k):
    '''
    returns the boundary operator of order k on n vertices as a dictionary
    '''
    dictionary = {}
    dictionary[
        (
            tuple([0]*n_vertices),
            tuple([0]*n_vertices)
            )
        ] = 0
    for bound in it.product(range(2), repeat=n_vertices):
        if np.sum(bound) == k+1:
            helper = boundary_operator(bound)
            for key in helper:
                dictionary[(tuple(bound), key)] = helper[key]
    return dictionary


def boundary_operator_crsmat(n_vertices):
    '''
    returns boundary operator on n vertices as sparse crs matrix
    '''
    dictionary = boundary_operator_dict(n_vertices)
    basis_list = list(it.product(range(2), repeat=n_vertices))
    basis_dict = {basis_list[i]: i for i in range(2**n_vertices)}
    col = np.array([basis_dict[index[0]] for index in dictionary])
    row = np.array([basis_dict[index[1]] for index in dictionary])
    data = np.array(list(dictionary.values()))
    return csr_matrix((data, (row, col)), shape=(2**n_vertices, 2**n_vertices))


def boundary_operator_crsmat_k(n_vertices, k):
    '''
    returns boundary operator of order k on n vertices as sparse crs matrix
    '''
    dictionary = boundary_operator_dict_k(n_vertices, k)
    basis_list = list(it.product(range(2), repeat=n_vertices))
    basis_dict = {basis_list[i]: i for i in range(2**n_vertices)}
    col = np.array([basis_dict[index[0]] for index in dictionary])
    row = np.array([basis_dict[index[1]] for index in dictionary])
    data = np.array(list(dictionary.values()))
    return csr_matrix((data, (row, col)), shape=(2**n_vertices, 2**n_vertices))


def combinatorial_laplacian(n_vertices, k):
    '''
    returns combinatorial Laplacian of order k on n vertices
    as sparse crs matrix
    '''
    delta_k = boundary_operator_crsmat_k(n_vertices, k)
    delta_kplus1 = boundary_operator_crsmat_k(n_vertices, k+1)
    return delta_k.conj().T @ delta_k + delta_kplus1 @ delta_kplus1.conj().T


def projector_onto_state(n_vertices, state):
    '''projector onto simplex state'''
    basis_list = list(it.product(range(2), repeat=n_vertices))
    basis_dict = {basis_list[i]: i for i in range(2**n_vertices)}
    indices = [basis_dict[s] for s in state]
    return csr_matrix(
        (np.ones(len(indices)), (indices, indices)),
        shape=(2**n_vertices, 2**n_vertices)
        )


def projected_combinatorial_laplacian(n_vertices, k, state_dict):
    '''
    returns the projected combinatorial Laplacian of order k on n vertices
    as sparse crs matrix
    '''
    p_k = projector_onto_state(n_vertices, state_dict[k])
    p_kp1 = projector_onto_state(n_vertices, state_dict[k+1])
    delta_k = boundary_operator_crsmat_k(n_vertices, k) @ p_k
    delta_kplus1 = boundary_operator_crsmat_k(n_vertices, k+1) @ p_kp1
    return delta_k.conj().T @ delta_k + delta_kplus1 @ delta_kplus1.conj().T


def initialize_projector(
        state,
        circuit=None,
        initialization_qubits=None,
        circuit_name=None
        ):
    '''
    initializes projector onto subspace spanned by list of states
    input circuit has to have classical register
    '''
    if circuit is None:
        n_vertices = len(state[0])
        qr1 = QuantumRegister(n_vertices, name="state_reg")
        copy_reg = QuantumRegister(n_vertices, name="copy_reg")
        quantum_circuit = QuantumCircuit(qr1, copy_reg)
        state_vec = state_to_vec(state)
        state_vec = state_vec/np.linalg.norm(state_vec)
        quantum_circuit.initialize(state_vec, qr1)
        # quantum_circuit.barrier()
        for k in range(n_vertices):
            quantum_circuit.cx(qr1[k], copy_reg[k])
        # quantum_circuit.barrier()
    else:
        n_vertices = len(state[0])
        qr1 = QuantumRegister(n_vertices, name="state_reg")
        copy_reg = QuantumRegister(n_vertices, name="copy_reg")
        if circuit.num_qubits - n_vertices > 0:
            anr = QuantumRegister(
                circuit.num_qubits - n_vertices,
                name="an_reg"
                )
            quantum_circuit = QuantumCircuit(anr, qr1, copy_reg)
        else:
            quantum_circuit = QuantumCircuit(qr1, copy_reg)
        state_vec = state_to_vec(state)
        state_vec = state_vec/np.linalg.norm(state_vec)
        quantum_circuit.initialize(state_vec, qr1)
        # quantum_circuit.barrier()
        for k in range(n_vertices):
            quantum_circuit.cx(qr1[k], copy_reg[k])
        # quantum_circuit.barrier()
        if initialization_qubits is None:
            init = list(range(n_vertices))
        else:
            init = initialization_qubits
        rest = list(set(range(circuit.num_qubits)) - set(init))
        sub_inst = circuit.to_instruction()
        if circuit_name is not None:
            sub_inst.name = circuit_name
        quantum_circuit.append(sub_inst, rest + init)
    return quantum_circuit


def simplices_to_states(test_list, n_vertices):
    '''
    converts the simplices to a matrix
    '''
    n_states = len(test_list)
    arr = np.zeros((n_states, n_vertices))
    for k in range(n_states):
        arr[k, test_list[k]] = 1
    return arr


class DataFiltration:
    '''
    data: point data given by (number_data_points x dim_points)-numpy-array
    distance_matrix: (n x n)-numpy-array describing the pair-wise
        distances of n data points
    Either one of them has to be given!
    '''
    def __init__(
            self,
            data=None,
            distance_matrix=None,
            max_dimension=None,
            max_edge_length=None
            ):
        if data is not None:
            self.skeleton = gd.RipsComplex(
                points=data,
                max_edge_length=max_edge_length
                )
        elif distance_matrix is not None:
            self.skeleton = gd.RipsComplex(
                distance_matrix=distance_matrix,
                max_edge_length=max_edge_length
                )
        else:
            print('Either point data or distance matrix has to be provided!')

        self.Rips_simplex_tree = self.skeleton.create_simplex_tree(
            max_dimension=max_dimension
            )
        self.num_vertices = self.Rips_simplex_tree.num_vertices()
        self.num_simplices = self.Rips_simplex_tree.num_simplices()
        self.filtration = list(self.Rips_simplex_tree.get_filtration())

    def get_filtration_states(self, epsilons=None):
        '''
        epsilons: different radia for the filtration
        returns a dictionary of the filtration
        '''
        if epsilons is None:
            epsilons = set([x[1] for x in self.filtration])
        filt_dict = {}
        for eps in epsilons:
            helper_list = [x[0] for x in self.filtration if x[1] <= eps]
            filt_dict[eps] = simplices_to_states(
                helper_list, self.num_vertices
                )
        return filt_dict

    def plot_persistence_diagram(self):
        ''' plots a diagram for the persistent topologial features '''
        bar_codes = self.Rips_simplex_tree.persistence()
        gd.plot_persistence_diagram(bar_codes, legend=True)


def controled_u(unitary, quantum_circuit, num_eval_qubits, n_vertices):  # , k, state_dict):
    '''
    returns a quantum circuit with the controled unitary for the
    quantum phase estimation routine.
    '''
    unit = unitary
    gate = UnitaryGate(unit)
    for counting_qubit in range(num_eval_qubits):
        quantum_circuit.append(
            gate.control(1),
            [counting_qubit] + list(
                range(num_eval_qubits, num_eval_qubits+n_vertices)
                )
            )
        unit = unit@unit
        gate = UnitaryGate(unit)
    return quantum_circuit


def qft_dagger(quantum_circuit, num):
    """n-qubit QFTdagger the first n qubits in circ"""
    # Don't forget the Swaps!
    for qubit in range(num//2):
        quantum_circuit.swap(qubit, num-qubit-1)
    for j in range(num):
        for k in range(j):
            quantum_circuit.cp(-np.pi/float(2**(j-k)), k, j)
        quantum_circuit.h(j)


def qpe_total(num_eval_qubits, n_vertices, unitary):  # , k, state_dict):
    '''
    returns the full circuit of the quantum phase estimation
    '''
    quantum_circuit = QuantumCircuit(num_eval_qubits + n_vertices)
    for qubit in range(num_eval_qubits):
        quantum_circuit.h(qubit)
    controled_u(unitary, quantum_circuit, num_eval_qubits, n_vertices)  # , k, state_dict)
    # Apply inverse QFT
    qft_dagger(quantum_circuit, num_eval_qubits)
    return quantum_circuit


class QTDAalgorithm(QuantumCircuit):
    '''
    Class that contains the QTDA algorithm.
    num_eval_qubits: number of qubits for the evaluation in the quantum phase
                     estimation
    top_order: the topological order
    state_dict: the dictionary of states
    '''
    def __init__(self, num_eval_qubits, top_order, state_dict, name='QTDA'):
        n_vertices = len(state_dict[0][0])
        qr_eval = QuantumRegister(num_eval_qubits, 'eval')
        qr_state = QuantumRegister(n_vertices, 'state')
        qr_copy = QuantumRegister(n_vertices, 'copy')
        super().__init__(qr_eval, qr_state, qr_copy, name=name)

        unitary = expm(
            1j*projected_combinatorial_laplacian(
                n_vertices, top_order, state_dict
                ).toarray()
            )

        '''
        we can also directly use the PhaseEstimation routine from Qiskit,
        which, however, leads to a larger circuit depth than our designed
        qpe_total implementation
        '''
        # gate = UnitaryGate(unitary)
        # qpe = qiskit.circuit.library.PhaseEstimation(
        #     num_eval_qubits, unitary=gate, iqft=None, name='QPE'
        #     )
        qpe = qpe_total(num_eval_qubits, n_vertices, unitary)
        sub_inst = qpe.to_instruction()
        sub_inst.name = '        QPE        '

        state_vec = state_to_vec(state_dict[top_order])
        state_vec = state_vec/np.linalg.norm(state_vec)
        self.initialize(state_vec, qr_state)
        # for a better visualisation one can imput barriers, but they slow
        # down the computation.
        # self.barrier()
        for k in range(n_vertices):
            self.cx(qr_state[k], qr_copy[k])
        # self.barrier()
        self.append(sub_inst, list(range(num_eval_qubits + n_vertices)))
        self.eval_qubits = list(range(num_eval_qubits))


class Q_top_spectra:
    '''
    data: point data given by (number_data_points x dim_points)-numpy-array
    distance_matrix: (n x n)-numpy-array describing the pair-wise
        distances of n data points
    Either one of them has to be given!
    '''
    def __init__(self, state_dict, num_eval_qubits=6, shots=1000):
        self.shots = shots
        self.state_dict = state_dict
        self.counts = {}
        for top_order in self.state_dict:
            print('Topological order: ', top_order)
            if len(self.state_dict[top_order]) == 0:
                print(
                    "calculation terminated because no simplex of dimension %s" % (top_order)
                    )
                break
            quantum_circuit = QTDAalgorithm(num_eval_qubits, top_order, self.state_dict)
            quantum_circuit.add_register(ClassicalRegister(num_eval_qubits, name="phase"))
            for qubit in quantum_circuit.eval_qubits:
                quantum_circuit.measure(qubit, qubit)
            backend = Aer.get_backend('qasm_simulator')
            job = execute(quantum_circuit, backend, shots=self.shots)
            self.counts[top_order] = job.result().get_counts(quantum_circuit)

    def get_counts(self):
        '''
        returns the number of counts
        '''
        return self.counts

    def get_spectra(self, chop=None):
        '''
        defaul chop: eigenvalues with less then 2% of all shots are regarded
            as noise (can be adjusted)
        exception: the eigenspace of eigenvalue 0 is important, even if it is
            0-dimensional; hence, this is always included
        '''
        if chop is None:
            chop = self.shots/50
        eigenvalue_dict = {}
        for top_order in self.counts:
            eigenvalue_dict[top_order] = {}
            eigenvalue_dict[top_order][0.0] = 0
            vals = np.fromiter(self.counts[top_order].values(), dtype=float)
            keys = list(self.counts[top_order].keys())
            indices = np.where(vals >= chop)
            new_vals = vals[indices]
            new_keys = [keys[i] for i in indices[0]]
            for i in range(len(new_keys)):
                eigenvalue_dict[top_order][
                    2*np.pi*int(new_keys[i], 2)/int(len(new_keys[i])*'1', 2)
                    ] = (
                    new_vals[i]
                    * len(self.state_dict[top_order])
                    / self.shots
                    )
        return eigenvalue_dict


class Q_persistent_top_spectra:
    '''
    data: point data given by (number_data_points x dim_points)-numpy-array
    distance_matrix: (n x n)-numpy-array describing the pair-wise
        distances of n data points
    Either one of them has to be given!
    '''
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

        self.shots = shots
        self.filt_dict = DataFiltration(
            data=data,
            distance_matrix=distance_matrix,
            max_dimension=max_dimension,
            max_edge_length=max_edge_length
            ).get_filtration_states(epsilons=epsilons)
        self.state_dict = {}
        for key in sorted(self.filt_dict.keys()):
            self.state_dict[key] = {}
            for k in range(1, max_dimension+1):
                mask = np.sum(self.filt_dict[key], axis=1) == k
                self.state_dict[key][k-1] = [
                    tuple(s)
                    for s in self.filt_dict[key][mask, :]
                    ]
            self.state_dict[key][max_dimension] = []  # an empty state has to be included
            # on order max_dimension for consistency

        self.counts = {}
        for eps in self.state_dict:
            print()
            print('Filtration scale: ', eps)
            print()
            self.counts[eps] = {}
            for top_order in self.state_dict[eps]:
                print('Topological order: ', top_order)
                if len(self.state_dict[eps][top_order]) == 0:
                    print(
                        "calculation terminated because no simplex of dimension %s" % (top_order)
                        )
                    break

                quantum_circuit = QTDAalgorithm(
                    num_eval_qubits,
                    top_order,
                    self.state_dict[eps]
                    )
                quantum_circuit.add_register(
                    ClassicalRegister(num_eval_qubits, name="phase")
                    )
                for qubit in quantum_circuit.eval_qubits:
                    quantum_circuit.measure(qubit, qubit)
                backend = Aer.get_backend('qasm_simulator')
                job = execute(quantum_circuit, backend, shots=self.shots)
                self.counts[eps][top_order] = job.result().get_counts(quantum_circuit)

    def get_counts(self):
        '''
        returns the number of counts
        '''
        return self.counts

    def get_eigenvalues(self, chop=None):
        '''
        defaul chop: eigenvalues with less then 2% of all shots are regarded
            as noise (can be adjusted)
        exception: the eigenspace of eigenvalue 0 is important, even if it is
            0-dimensional; hence, this is always included
        '''
        if chop is None:
            chop = self.shots/50

        eigenvalue_dict = {}
        for eps in self.counts:
            eigenvalue_dict[eps] = {}
            for top_order in self.counts[eps]:
                eigenvalue_dict[eps][top_order] = {}
                eigenvalue_dict[eps][top_order][0.0] = 0
                vals = np.fromiter(
                    self.counts[eps][top_order].values(), dtype=float
                    )
                keys = list(self.counts[eps][top_order].keys())
                indices = np.where(vals >= chop)
                new_vals = vals[indices]
                new_keys = [keys[i] for i in indices[0]]
                for i, key in enumerate(new_keys):
                    eigenvalue_dict[eps][top_order][
                        2*np.pi*int(key, 2)/int(len(new_keys[i])*'1', 2)
                        ] = (
                            new_vals[i]
                            * len(self.state_dict[eps][top_order])
                            / self.shots
                            )
        return eigenvalue_dict
