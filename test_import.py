# Save as E:\VI\test_fidelity.py
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.circuit.library import PauliFeatureMap
from qiskit_aer import AerSimulator
feature_map = PauliFeatureMap(2, reps=1)
kernel = FidelityQuantumKernel(feature_map=feature_map)
print("FidelityQuantumKernel initialized successfully")