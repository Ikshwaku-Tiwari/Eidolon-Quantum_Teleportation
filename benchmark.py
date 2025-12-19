import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit.quantum_info import DensityMatrix, partial_trace, state_fidelity
from qiskit.transpiler import Target, InstructionProperties
from qiskit.circuit.library import XGate, SXGate, RZGate, CXGate, SwapGate, UGate, Measure
from qiskit.circuit.controlflow import IfElseOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import warnings

noise_level = 0.02 
warnings.filterwarnings("ignore")

print("🧪 Starting Eídōlon Benchmark Experiment...")
print(f"   Target Topology: Star Graph (Disjoint q1-q2)")
print(f"   Noise Level: {noise_level}")

target = Target(num_qubits=3) 

single_q_props = {(i,): None for i in range(3)}
target.add_instruction(UGate(theta=0, phi=0, lam=0), single_q_props)
target.add_instruction(XGate(), single_q_props)
target.add_instruction(SXGate(), single_q_props)
target.add_instruction(RZGate(0), single_q_props)
target.add_instruction(Measure(), single_q_props)

target.add_instruction(IfElseOp, name="if_else")

edges = [(0, 1), (1, 0), (0, 2), (2, 0)]
two_q_props = {edge: None for edge in edges}
target.add_instruction(CXGate(), two_q_props)
target.add_instruction(SwapGate(), two_q_props)

noise_model = NoiseModel()
error = depolarizing_error(noise_level, 2)
noise_model.add_all_qubit_quantum_error(error, ['cx', 'swap'])
simulator = AerSimulator(noise_model=noise_model, method='density_matrix')

theta, phi = np.pi/2, 0.0
qr = QuantumRegister(3, 'q')
cr_z = ClassicalRegister(1, 'cz')
cr_x = ClassicalRegister(1, 'cx')
qc = QuantumCircuit(qr, cr_z, cr_x)

qc.u(theta, phi, 0, 0)
qc.h(1)
qc.cx(1, 2)
qc.cx(0, 1)
qc.h(0)
qc.measure([0, 1], [0, 1])

with qc.if_test((cr_x, 1)): qc.x(2)
with qc.if_test((cr_z, 1)): qc.z(2)

qc_ideal = QuantumCircuit(3)
qc_ideal.u(theta, phi, 0, 0)
ideal_rho = partial_trace(DensityMatrix(qc_ideal), [1, 2])

levels = [1, 2, 3] 
results = []
print("-" * 50)

for lvl in levels:
    print(f"   Running Optimization Level {lvl}...")
    
    t_qc = transpile(qc, target=target, optimization_level=lvl)
    
    t_qc_sim = t_qc.copy()
    t_qc_sim.save_density_matrix(label='rho')
    
    result = simulator.run(t_qc_sim).result()
    if 'rho' in result.data():
        rho_out = result.data()['rho']
        rho_q2 = partial_trace(rho_out, [0, 1])
        fid = state_fidelity(ideal_rho, rho_q2)
    else:
        fid = 0.0
    
    results.append({
        "Opt Level": f"Level {lvl}",
        "Fidelity": fid,
        "SWAP Count": t_qc.count_ops().get('swap', 0),
        "Depth": t_qc.depth()
    })

df = pd.DataFrame(results)
print("-" * 50)
print("RESULTS TABLE:")
print(df)
df.to_csv("benchmark_results.csv", index=False)

fig, ax1 = plt.subplots(figsize=(8, 5))
color = 'tab:blue'
ax1.set_xlabel('Optimization Level')
ax1.set_ylabel('Fidelity', color=color)
bars = ax1.bar(df['Opt Level'], df['Fidelity'], color=color, alpha=0.6)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, 1.05)

for bar in bars:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), round(bar.get_height(), 3), va='bottom', ha='center')

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('SWAP Gates', color=color)  
ax2.plot(df['Opt Level'], df['SWAP Count'], color=color, marker='o', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Optimization Impact on Teleportation Fidelity')
fig.tight_layout()
plt.savefig('benchmark_plot.png')
print("\n✅ Plot saved as 'benchmark_plot.png'")

#The next update will include adding "Trash gates" to see the fidelity drop for level1