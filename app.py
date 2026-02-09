import streamlit as st
import numpy as np
import plotly.graph_objects as go
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit.quantum_info import DensityMatrix, partial_trace, Operator, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import matplotlib.pyplot as plt

PAULI_X = Operator([[0, 1], [1, 0]])
PAULI_Y = Operator([[0, -1j], [1j, 0]])
PAULI_Z = Operator([[1, 0], [0, -1]])

def get_bloch_coordinates(quantum_state, q_index):
    num_qubits = quantum_state.num_qubits
    qubits_to_trace_out = list(range(num_qubits))
    qubits_to_trace_out.pop(q_index)
    rho_qubit = partial_trace(quantum_state, qubits_to_trace_out)
    x = rho_qubit.expectation_value(PAULI_X).real
    y = rho_qubit.expectation_value(PAULI_Y).real
    z = rho_qubit.expectation_value(PAULI_Z).real
    return [x, y, z]

def get_purity(quantum_state, q_index):
    num_qubits = quantum_state.num_qubits
    qubits_to_trace_out = list(range(num_qubits))
    qubits_to_trace_out.pop(q_index)
    rho_qubit = partial_trace(quantum_state, qubits_to_trace_out)
    return rho_qubit.purity().real

def create_bloch_sphere(vector, title):
    sphere = go.Surface(
        x=np.outer(np.cos(np.linspace(0, 2 * np.pi, 30)), np.sin(np.linspace(0, np.pi, 30))),
        y=np.outer(np.sin(np.linspace(0, 2 * np.pi, 30)), np.sin(np.linspace(0, np.pi, 30))),
        z=np.outer(np.ones(30), np.cos(np.linspace(0, np.pi, 30))),
        opacity=0.2,
        showscale=False,
        colorscale=[[0, 'rgb(100,100,100)'], [1, 'rgb(100,100,100)']],
        name="Sphere"
    )
    arrow = go.Scatter3d(
        x=[0, vector[0]],
        y=[0, vector[1]],
        z=[0, vector[2]],
        mode='lines',
        line=dict(color='red', width=10),
        name='State Vector'
    )
    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X', range=[-1, 1], showticklabels=False, showgrid=False),
            yaxis=dict(title='Y', range=[-1, 1], showticklabels=False, showgrid=False),
            zaxis=dict(title='Z (|0⟩ / |1⟩)', range=[-1, 1], showticklabels=True),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig = go.Figure(data=[sphere, arrow], layout=layout)
    return fig

st.set_page_config(layout="wide")
st.title("Eídōlon (The Phantom)")
st.markdown("An interactive 3-step visualization of the 2022 Nobel Prize-winning quantum teleportation protocol.")

col_a, col_b = st.columns(2)
with col_a:
    st.header(f"1. Create Orpheus's 'Message' Qubit ($q_0$)")
    theta_input = st.slider(
        "Set Latitude (θ):",
        min_value=0.0,
        max_value=np.pi,
        value=np.pi / 2,
        format="%.2f rad"
    )
    phi_input = st.slider(
        "Set Longitude/Phase (φ):",
        min_value=0.0,
        max_value=2 * np.pi,
        value=0.0,
        format="%.2f rad"
    )
    st.latex(rf"|\psi\rangle = \cos({theta_input/2:.2f})|0\rangle + e^{{i{phi_input:.2f}}}\sin({theta_input/2:.2f})|1\rangle")

with col_b:
    st.header("2. Set Simulation Noise")
    noise_level = st.slider(
        "Set 2-Qubit Gate Error Rate (Depolarizing Noise):",
        min_value=0.0,
        max_value=0.1,
        value=0.0,
        format="%.3f"
    )
    st.markdown("This error is applied to all `cx` and `swap` gates.")

coupling_map = [[0, 1], [1, 0], [1, 2], [2, 1]]
basis_gates = ['id', 'rz', 'sx', 'x', 'cx', 'swap', 'u', 'h']

noise_model = NoiseModel()
if noise_level > 0:
    error = depolarizing_error(noise_level, 2)
    noise_model.add_all_qubit_quantum_error(error, ['cx', 'swap'])

simulator = AerSimulator(noise_model=noise_model)

st.divider()
st.header("3. The Simulation")
st.markdown("Follow the state of the three qubits from left to right.")

col1, col2, col3 = st.columns(3)

ideal_rho_q0 = None

with col1:
    st.subheader("Step 1: Initial State")
    before_qc = QuantumCircuit(3)
    before_qc.u(theta_input, phi_input, 0, 0)
    before_dm = DensityMatrix(before_qc)
    
    q0_coords = get_bloch_coordinates(before_dm, 0)
    q1_coords = get_bloch_coordinates(before_dm, 1)
    q2_coords = get_bloch_coordinates(before_dm, 2)
    
    ideal_rho_q0 = partial_trace(before_dm, [1, 2])
    
    st.plotly_chart(create_bloch_sphere(q0_coords, "q₀ (Orpheus's Message)"), width='stretch')
    st.metric("Purity", f"{get_purity(before_dm, 0):.3f}")
    st.plotly_chart(create_bloch_sphere(q1_coords, "q₁ (Orpheus's Link)"), width='stretch')
    st.metric("Purity", f"{get_purity(before_dm, 1):.3f}")
    st.plotly_chart(create_bloch_sphere(q2_coords, "q₂ (Eurydice's Qubit)"), width='stretch')
    st.metric("Purity", f"{get_purity(before_dm, 2):.3f}")
    st.caption("Orpheus has his message ($q_0$). $q_1$ and $q_2$ are blank.")

with col2:
    st.subheader("Step 2: Just Before Measurement")
    mid_qc = QuantumCircuit(3)
    mid_qc.u(theta_input, phi_input, 0, 0)
    mid_qc.h(1)
    mid_qc.cx(1, 2)
    mid_qc.barrier()
    mid_qc.cx(0, 1)
    mid_qc.h(0)
    
    transpiled_mid_qc = transpile(mid_qc, coupling_map=coupling_map, basis_gates=basis_gates, optimization_level=0)
    transpiled_mid_qc.save_density_matrix(label="mid_dm")
    
    result_mid = simulator.run(transpiled_mid_qc).result()
    mid_dm = result_mid.data()["mid_dm"]
    
    q0_coords = get_bloch_coordinates(mid_dm, 0)
    q1_coords = get_bloch_coordinates(mid_dm, 1)
    q2_coords = get_bloch_coordinates(mid_dm, 2)
    
    st.plotly_chart(create_bloch_sphere(q0_coords, "q₀ (Orpheus's Qubit)"), width='stretch')
    st.metric("Purity", f"{get_purity(mid_dm, 0):.3f}")
    st.plotly_chart(create_bloch_sphere(q1_coords, "q₁ (Orpheus's Link)"), width='stretch')
    st.metric("Purity", f"{get_purity(mid_dm, 1):.3f}")
    st.plotly_chart(create_bloch_sphere(q2_coords, "q₂ (Eurydice's Qubit)"), width='stretch')
    st.metric("Purity", f"{get_purity(mid_dm, 2):.3f}")
    st.caption("Orpheus performs his operations, entangling all 3 qubits. The states of $q_0$ and $q_1$ are 'destroyed' (mixed) and $q_2$ is in a 'broken' state.")

with col3:
    st.subheader("Step 3: Eurydice Corrects")
    
    proj_0 = np.array([[1, 0], [0, 0]])
    proj_1 = np.array([[0, 0], [0, 1]])
    I2 = np.eye(2)
    
    projectors = {
        (0, 0): np.kron(I2, np.kron(proj_0, proj_0)),
        (0, 1): np.kron(I2, np.kron(proj_1, proj_0)),
        (1, 0): np.kron(I2, np.kron(proj_0, proj_1)),
        (1, 1): np.kron(I2, np.kron(proj_1, proj_1)),
    }
    
    X_gate = np.array([[0, 1], [1, 0]])
    Z_gate = np.array([[1, 0], [0, -1]])
    
    corrections = {
        (0, 0): np.eye(8),
        (0, 1): np.kron(X_gate, np.kron(I2, I2)),
        (1, 0): np.kron(Z_gate, np.kron(I2, I2)),
        (1, 1): np.kron(Z_gate @ X_gate, np.kron(I2, I2)),
    }
    
    rho = mid_dm.data
    
    accumulated_dm = np.zeros((8, 8), dtype=complex)
    
    for (m0, m1), P in projectors.items():
        projected_rho = P @ rho @ P.conj().T
        prob = np.trace(projected_rho).real
        
        if prob > 1e-10:
            normalized_rho = projected_rho / prob
            C = corrections[(m0, m1)]
            corrected_rho = C @ normalized_rho @ C.conj().T
            accumulated_dm += prob * corrected_rho
    
    final_dm = DensityMatrix(accumulated_dm)
    
    q0_coords = get_bloch_coordinates(final_dm, 0)
    q1_coords = get_bloch_coordinates(final_dm, 1)
    q2_coords = get_bloch_coordinates(final_dm, 2)
    
    st.plotly_chart(create_bloch_sphere(q0_coords, "q₀ (Orpheus's Spent Qubit)"), width='stretch')
    st.metric("q₀ Purity", f"{get_purity(final_dm, 0):.3f}")
    
    st.plotly_chart(create_bloch_sphere(q1_coords, "q₁ (Orpheus's Spent Link)"), width='stretch')
    st.metric("q₁ Purity", f"{get_purity(final_dm, 1):.3f}")
    
    st.plotly_chart(create_bloch_sphere(q2_coords, "q₂ (Eurydice's Recovered State)"), width='stretch')
    
    final_rho_q2 = partial_trace(final_dm, [0, 1])
    fidelity = state_fidelity(ideal_rho_q0, final_rho_q2)
    
    st.metric("q₂ Purity", f"{get_purity(final_dm, 2):.3f}")
    st.metric("Final Fidelity", f"{fidelity:.3f}")
    
    st.caption("Eurydice gets Orpheus's 2 classical bits, applies her correction, and recovers the original message. Teleportation complete!")

st.divider()
st.header("4. The Full Quantum Circuit")
st.markdown("This is the 'ideal' circuit. It's clean and simple.")

display_qc = QuantumCircuit(QuantumRegister(3, "q"), ClassicalRegister(2, "c"))
display_qc.u(theta_input, phi_input, 0, 0)
display_qc.h(1)
display_qc.cx(1, 2)
display_qc.barrier()
display_qc.cx(0, 1)
display_qc.h(0)
display_qc.barrier()
display_qc.measure([0, 1], [0, 1])
display_qc.barrier()
with display_qc.if_test((display_qc.cregs[0][1], 1)):
    display_qc.x(2)
with display_qc.if_test((display_qc.cregs[0][0], 1)):
    display_qc.z(2)

fig_ideal, ax_ideal = plt.subplots(figsize=(12, 4))
display_qc.draw(output='mpl', ax=ax_ideal, fold=-1)
st.pyplot(fig_ideal)

st.markdown("---")
st.markdown("This is the 'real' circuit the transpiler runs on the hardware. **Notice the 'swap' gates** added to manage the bad hardware layout.")

display_for_transpile = QuantumCircuit(3)
display_for_transpile.u(theta_input, phi_input, 0, 0)
display_for_transpile.h(1)
display_for_transpile.cx(1, 2)
display_for_transpile.cx(0, 1)
display_for_transpile.h(0)

transpiled_display_qc = transpile(display_for_transpile, coupling_map=coupling_map, basis_gates=basis_gates, optimization_level=0)

fig_real, ax_real = plt.subplots(figsize=(14, 4))
transpiled_display_qc.draw(output='mpl', ax=ax_real, idle_wires=False, fold=-1)
st.pyplot(fig_real)