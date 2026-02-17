#!/usr/bin/env python3
"""
LiH Molecular Hamiltonian Comparison with Z2 Symmetry Tapering
===============================================================
Project Molecular Symphony - Phase 4, Week 1 Refinement

Compares second-quantized Hamiltonians of Lithium Hydride (LiH) using
Jordan-Wigner and Bravyi-Kitaev fermion-to-qubit mappings, with Z2
symmetry-based qubit tapering for circuit depth reduction.

Author: Ikshwaku Tiwari
Date: 2026-01-27
"""

from __future__ import annotations

import pickle
from typing import Tuple, Dict, Any, List

import jax

jax.config.update("jax_platforms", "cpu")

import jax.numpy as jnp
import numpy as np
import pennylane as qml
from pennylane import qchem

SYMBOLS = ["Li", "H"]
GEOMETRY = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.59],
    ]
)

BASIS_SET = "sto-3g"
CHARGE = 0
MULTIPLICITY = 1
ELECTRONS = 4

SUPPORTED_MAPPINGS = ["jordan_wigner", "bravyi_kitaev"]


def build_molecular_hamiltonian(
    mapping: str = "jordan_wigner",
) -> Tuple[qml.Hamiltonian, int]:
    """
    Generate the molecular Hamiltonian for LiH.

    Args:
        mapping: Fermion-to-qubit transformation ('jordan_wigner' or 'bravyi_kitaev')

    Returns:
        Tuple of (Hamiltonian operator, number of qubits)
    """
    if mapping not in SUPPORTED_MAPPINGS:
        raise ValueError(
            f"Unsupported mapping: {mapping}. Use one of {SUPPORTED_MAPPINGS}"
        )

    geometry_np = GEOMETRY.flatten()

    hamiltonian, qubits = qchem.molecular_hamiltonian(
        symbols=SYMBOLS,
        coordinates=geometry_np,
        basis=BASIS_SET,
        charge=CHARGE,
        mult=MULTIPLICITY,
        mapping=mapping,
        method="pyscf",
    )

    return hamiltonian, qubits


def get_hamiltonian(mapping_type: str) -> qml.Hamiltonian:
    """Get the molecular Hamiltonian for a specific mapping."""
    hamiltonian, _ = build_molecular_hamiltonian(mapping_type)
    return hamiltonian


def count_pauli_terms(hamiltonian) -> int:
    """Count the total number of Pauli string terms in the Hamiltonian."""
    if hasattr(hamiltonian, "ops"):
        return len(hamiltonian.ops)
    elif hasattr(hamiltonian, "operands"):
        return len(hamiltonian.operands)
    return 0


def get_pauli_weight(pauli_term) -> int:
    """
    Calculate the Pauli weight of a single term.
    Pauli weight = number of non-Identity operators in the Pauli string.
    """
    if isinstance(pauli_term, qml.Identity):
        return 0

    if hasattr(pauli_term, "operands"):
        return sum(1 for op in pauli_term.operands if not isinstance(op, qml.Identity))
    elif hasattr(pauli_term, "wires"):
        return len(pauli_term.wires)

    return 0


def get_max_pauli_weight(hamiltonian) -> int:
    """Find the maximum Pauli weight across all terms in the Hamiltonian."""
    ops = hamiltonian.ops if hasattr(hamiltonian, "ops") else []
    if hasattr(hamiltonian, "operands"):
        ops = hamiltonian.operands

    if not ops:
        return 0

    return max(get_pauli_weight(term) for term in ops)


def get_symmetry_generators(hamiltonian) -> List:
    """
    Find Z2 symmetry generators of the Hamiltonian.

    Args:
        hamiltonian: PennyLane Hamiltonian object

    Returns:
        List of Pauli word generators representing Z2 symmetries
    """
    generators = qml.symmetry_generators(hamiltonian)
    return generators


def get_optimal_sector(generators: List, n_electrons: int, n_qubits: int) -> List[int]:
    """
    Find the optimal parity sector for the given number of electrons.

    Args:
        generators: List of Z2 symmetry generators
        n_electrons: Number of electrons in the molecule
        n_qubits: Number of qubits in the system

    Returns:
        List of eigenvalues (+1 or -1) defining the sector
    """
    hf_state = qml.qchem.hf_state(n_electrons, n_qubits)

    paulixops = qml.paulix_ops(generators, n_qubits)

    sector = qml.qchem.optimal_sector(
        hamiltonian=None, generators=generators, active_electrons=n_electrons
    )

    return sector


def taper_hamiltonian(
    hamiltonian, generators: List, sector: List[int], paulixops: List
):
    """
    Apply Z2 symmetry tapering to reduce qubit count.

    Args:
        hamiltonian: Original Hamiltonian
        generators: Z2 symmetry generators
        sector: Parity sector eigenvalues
        paulixops: Pauli X operators for tapering

    Returns:
        Tapered Hamiltonian with reduced qubit count
    """
    tapered_h = qml.taper(hamiltonian, generators, paulixops, sector)
    return tapered_h


def analyze_hamiltonian_with_tapering(mapping: str) -> Dict[str, Any]:
    """
    Perform complete analysis including Z2 tapering for a given mapping.

    Args:
        mapping: Fermion-to-qubit mapping type

    Returns:
        Dictionary containing analysis results
    """
    hamiltonian, n_qubits = build_molecular_hamiltonian(mapping)

    original_terms = count_pauli_terms(hamiltonian)
    original_max_weight = get_max_pauli_weight(hamiltonian)

    generators = get_symmetry_generators(hamiltonian)
    n_symmetries = len(generators)

    tapered_h = None
    tapered_qubits = n_qubits
    tapered_terms = original_terms
    tapered_max_weight = original_max_weight

    if n_symmetries > 0:
        try:
            paulixops = qml.paulix_ops(generators, n_qubits)

            hf_state = qml.qchem.hf_state(ELECTRONS, n_qubits)
            sector = qml.qchem.optimal_sector(hamiltonian, generators, ELECTRONS)

            tapered_h = qml.taper(hamiltonian, generators, paulixops, sector)

            tapered_qubits = n_qubits - n_symmetries
            tapered_terms = count_pauli_terms(tapered_h)
            tapered_max_weight = get_max_pauli_weight(tapered_h)

        except Exception as e:
            print(f"   ⚠ Tapering warning: {e}")

    coefficients = jnp.array([float(c) for c in hamiltonian.coeffs])

    return {
        "mapping": mapping,
        "n_qubits": n_qubits,
        "n_terms": original_terms,
        "max_pauli_weight": original_max_weight,
        "n_symmetries": n_symmetries,
        "tapered_qubits": tapered_qubits,
        "tapered_terms": tapered_terms,
        "tapered_max_weight": tapered_max_weight,
        "coeffs_sum": float(jnp.sum(jnp.abs(coefficients))),
        "coeffs_max": float(jnp.max(jnp.abs(coefficients))),
        "hamiltonian": hamiltonian,
        "tapered_hamiltonian": tapered_h,
        "generators": generators,
    }


def save_tapered_hamiltonian(hamiltonian, filename: str) -> None:
    """Save the tapered Hamiltonian to a pickle file."""
    with open(filename, "wb") as f:
        pickle.dump(hamiltonian, f)
    print(f"   💾 Saved tapered Hamiltonian to {filename}")


def print_comparison_table(results: Dict[str, Dict[str, Any]]) -> None:
    """Print a formatted comparison table of mapping results with tapering."""
    print("\n" + "=" * 75)
    print("LiH MOLECULAR HAMILTONIAN COMPARISON (with Z2 Symmetry Tapering)")
    print("=" * 75)
    print(f"{'Metric':<35} {'Jordan-Wigner':>18} {'Bravyi-Kitaev':>18}")
    print("-" * 75)

    jw = results["jordan_wigner"]
    bk = results["bravyi_kitaev"]

    print(f"{'Original Qubits':<35} {jw['n_qubits']:>18} {bk['n_qubits']:>18}")
    print(f"{'Original Pauli Terms':<35} {jw['n_terms']:>18} {bk['n_terms']:>18}")
    print(
        f"{'Original Max Pauli Weight':<35} {jw['max_pauli_weight']:>18} {bk['max_pauli_weight']:>18}"
    )
    print("-" * 75)

    print(
        f"{'Z2 Symmetries Found':<35} {jw['n_symmetries']:>18} {bk['n_symmetries']:>18}"
    )
    print("-" * 75)

    print(
        f"{'Tapered Qubits':<35} {jw['tapered_qubits']:>18} {bk['tapered_qubits']:>18}"
    )
    print(
        f"{'Tapered Pauli Terms':<35} {jw['tapered_terms']:>18} {bk['tapered_terms']:>18}"
    )
    print(
        f"{'Tapered Max Pauli Weight':<35} {jw['tapered_max_weight']:>18} {bk['tapered_max_weight']:>18}"
    )
    print("=" * 75)

    jw_reduction = jw["n_qubits"] - jw["tapered_qubits"]
    bk_reduction = bk["n_qubits"] - bk["tapered_qubits"]
    print(f"{'Qubits Reduced':<35} {jw_reduction:>18} {bk_reduction:>18}")
    print("=" * 75)


def print_hardware_advantage(results: Dict[str, Dict[str, Any]]) -> None:
    """Explain hardware advantage for Phase 5 scaling."""
    bk = results["bravyi_kitaev"]

    original_dim = 2 ** bk["n_qubits"]
    tapered_dim = 2 ** bk["tapered_qubits"]

    print("\n📊 Hardware Advantage Analysis (Phase 5: 2^12 Scaling Goal)")
    print("-" * 60)
    print(f"   Original Hilbert space dimension: 2^{bk['n_qubits']} = {original_dim:,}")
    print(
        f"   Tapered Hilbert space dimension:  2^{bk['tapered_qubits']} = {tapered_dim:,}"
    )
    print(f"   Dimension reduction factor:       {original_dim / tapered_dim:.0f}x")
    print()

    if bk["tapered_qubits"] <= 10:
        print("   ✅ EXCELLENT: Tapered system fits within 2^10 = 1,024 dimensions")
        print("      → Classical simulation tractable for CVNN training")
        print("      → Can simulate full quantum dynamics on CPU")
        print()

    if bk["tapered_qubits"] <= 8:
        print("   🎯 OPTIMAL: 8-qubit system enables:")
        print("      → 256-dimensional statevector (manageable for JAX)")
        print("      → Efficient gradient computation for CVNN")
        print("      → Real-time variational optimization on Intel Iris Xe")


def main():
    """Main entry point for LiH Hamiltonian comparison with Z2 tapering."""
    print("\n" + "╔" + "═" * 73 + "╗")
    print(
        "║"
        + " LiH Molecular Hamiltonian Analysis with Z2 Symmetry Tapering ".center(73)
        + "║"
    )
    print("║" + " Project Molecular Symphony - Phase 4, Week 1 ".center(73) + "║")
    print("╚" + "═" * 73 + "╝")

    print(f"\n📊 Configuration:")
    print(f"   • Molecule: LiH ({ELECTRONS} electrons)")
    print(f"   • Geometry: Li at (0,0,0), H at (0,0,1.59) Å")
    print(f"   • Basis Set: {BASIS_SET}")
    print(f"   • Platform: JAX on {jax.default_backend().upper()}")

    results = {}

    for mapping in SUPPORTED_MAPPINGS:
        print(f"\n🔬 Analyzing {mapping.replace('_', ' ').title()} mapping...")
        results[mapping] = analyze_hamiltonian_with_tapering(mapping)
        r = results[mapping]
        print(f"   ✓ Original: {r['n_terms']} terms, {r['n_qubits']} qubits")
        print(f"   ✓ Symmetries found: {r['n_symmetries']}")
        print(f"   ✓ Tapered: {r['tapered_terms']} terms, {r['tapered_qubits']} qubits")

    print_comparison_table(results)
    print_hardware_advantage(results)

    bk_tapered = results["bravyi_kitaev"]["tapered_hamiltonian"]
    if bk_tapered is not None:
        save_tapered_hamiltonian(bk_tapered, "lih_h_tapered_bk.pkl")

    print("\n✅ Analysis complete.\n")

    return results


if __name__ == "__main__":
    results = main()
