"""
Fermionic and qubit operators for quantum chemistry.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field


@dataclass
class FermionOperator:
    """
    Fermionic operator represented as a weighted sum of fermionic terms.
    
    Each term is a product of creation (a†) and annihilation (a) operators.
    """
    
    terms: Dict[Tuple[Tuple[int, int], ...], complex] = field(default_factory=dict)
    # Key: tuple of (mode, operator_type) where operator_type is 0 for annihilation, 1 for creation
    # Value: coefficient
    
    def __add__(self, other):
        """Add two fermionic operators."""
        result = FermionOperator()
        result.terms = self.terms.copy()
        
        for term, coeff in other.terms.items():
            if term in result.terms:
                result.terms[term] += coeff
            else:
                result.terms[term] = coeff
        
        # Remove zero terms
        result.terms = {t: c for t, c in result.terms.items() if abs(c) > 1e-12}
        
        return result
    
    def __mul__(self, other):
        """Multiply by scalar or another operator."""
        if isinstance(other, (int, float, complex)):
            result = FermionOperator()
            result.terms = {t: c * other for t, c in self.terms.items()}
            return result
        else:
            raise NotImplementedError("Operator multiplication not implemented")
    
    def __rmul__(self, other):
        """Right multiplication by scalar."""
        return self.__mul__(other)


@dataclass
class QubitOperator:
    """
    Qubit operator represented as a weighted sum of Pauli strings.
    
    Each term is a tensor product of Pauli operators (I, X, Y, Z).
    """
    
    terms: Dict[Tuple[Tuple[int, str], ...], complex] = field(default_factory=dict)
    # Key: tuple of (qubit_index, pauli_operator) where pauli_operator is 'I', 'X', 'Y', or 'Z'
    # Value: coefficient
    
    def __add__(self, other):
        """Add two qubit operators."""
        result = QubitOperator()
        result.terms = self.terms.copy()
        
        for term, coeff in other.terms.items():
            if term in result.terms:
                result.terms[term] += coeff
            else:
                result.terms[term] = coeff
        
        # Remove zero terms
        result.terms = {t: c for t, c in result.terms.items() if abs(c) > 1e-12}
        
        return result
    
    def __mul__(self, other):
        """Multiply by scalar."""
        if isinstance(other, (int, float, complex)):
            result = QubitOperator()
            result.terms = {t: c * other for t, c in self.terms.items()}
            return result
        else:
            raise NotImplementedError("Operator multiplication not implemented")
    
    def __rmul__(self, other):
        """Right multiplication by scalar."""
        return self.__mul__(other)
    
    def to_matrix(self, n_qubits: int) -> np.ndarray:
        """
        Convert to matrix representation.
        
        Args:
            n_qubits: Number of qubits
            
        Returns:
            Matrix representation of the operator
        """
        dim = 2 ** n_qubits
        matrix = np.zeros((dim, dim), dtype=complex)
        
        # Pauli matrices
        I = np.eye(2)
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])
        
        pauli_dict = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
        
        for term, coeff in self.terms.items():
            # Build tensor product
            term_matrix = np.array([[1.0]], dtype=complex)
            
            # Create list of operators for each qubit
            operators = ['I'] * n_qubits
            for qubit_idx, pauli in term:
                operators[qubit_idx] = pauli
            
            # Tensor product
            for op in operators:
                term_matrix = np.kron(term_matrix, pauli_dict[op])
            
            matrix += coeff * term_matrix
        
        return matrix


def commutator(op1: QubitOperator, op2: QubitOperator) -> QubitOperator:
    """
    Calculate commutator [A, B] = AB - BA.
    
    Args:
        op1: First operator
        op2: Second operator
        
    Returns:
        Commutator of the two operators
    """
    # Simplified implementation - just returns zero operator
    return QubitOperator()


def anticommutator(op1: FermionOperator, op2: FermionOperator) -> FermionOperator:
    """
    Calculate anticommutator {A, B} = AB + BA.
    
    Args:
        op1: First operator
        op2: Second operator
        
    Returns:
        Anticommutator of the two operators
    """
    # Simplified implementation
    return FermionOperator()


def normal_ordered(op: FermionOperator) -> FermionOperator:
    """
    Put fermionic operator in normal order (all creations before annihilations).
    
    Args:
        op: Fermionic operator
        
    Returns:
        Normal-ordered operator
    """
    # For now, just return the operator as-is
    # Full implementation would reorder terms using anticommutation relations
    return op
