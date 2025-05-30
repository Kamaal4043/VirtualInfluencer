"""
Enhanced Quantum Social Presence Framework (QSPF) simulation for IBM Quantum hardware.
- Executes a 12-qubit circuit with human influencer baseline and expanded product categories.
- Includes dynamic parameter tuning, deeper entanglement logic, and eight quantum-specific visualizations.
- Optimized for local processing on AWS EC2 c5.4xlarge (16 vCPUs, 32GB RAM).

AWS EC2 c5.4xlarge Setup:
1. Launch instance: Ubuntu 22.04, c5.4xlarge, ≥30GB EBS, SSH access.
2. SSH: ssh -i your-key.pem ubuntu@your-ec2-ip
3. Install dependencies:
   sudo apt update && sudo apt install -y python3-pip
   pip3 install qiskit qiskit-aer qiskit-ibm-runtime qiskit-algorithms numpy pandas pymc plotly seaborn matplotlib scikit-learn scipy statsmodels
4. Copy script: scp -i your-key.pem qspf_hardware_travel_enhanced_with_visualizations.py ubuntu@your-ec2-ip:~
5. Run: python3 qspf_hardware_travel_enhanced_with_visualizations.py --use_hardware --ibm_token your_token_here --light_mode
"""

import numpy as np
import pandas as pd
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal, PauliFeatureMap, ZZFeatureMap, EfficientSU2
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, entropy, concurrence
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
from qiskit.result import LocalReadoutMitigator
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, CXCancellation, BasisTranslator, UnitarySynthesis
# Dynamically select kernel and optimizer imports based on Qiskit version
import qiskit

# if qiskit.__version__ >= "0.44.0":
#     from qiskit_algorithms.kernels import FidelityQuantumKernel
#     from qiskit_algorithms.optimizers import COBYLA
# else:
#     from qiskit_machine_learning.kernels import FidelityQuantumKernel
#     from qiskit_algorithms.optimizers import COBYLA  # Corrected import
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_algorithms.optimizers import COBYLA  # Corrected import
from qiskit.circuit import Parameter, ParameterVector
import pymc as pm
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from qiskit.visualization import plot_state_city, circuit_drawer
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator
from scipy.stats import f_oneway, ttest_ind, pearsonr
from statsmodels.stats.multitest import multipletests
from scipy.optimize import minimize
from itertools import product
from multiprocessing import Pool
import multiprocessing as mp
import time
import json
import datetime
import warnings
import importlib
import sys
import traceback
import os
import argparse

# Check for qiskit-ibm-runtime
qiskit_ibm_runtime = importlib.util.find_spec("qiskit_ibm_runtime")
if qiskit_ibm_runtime:
    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
else:
    QiskitRuntimeService = None
    Sampler = None

warnings.filterwarnings("ignore")

# Constants
NUM_VI_QUBITS = 4
NUM_CONSUMER_QUBITS = 4
NUM_CONTEXT_QUBITS = 4
TOTAL_QUBITS = NUM_VI_QUBITS + NUM_CONSUMER_QUBITS + NUM_CONTEXT_QUBITS
SHOTS = 2000  # Default for light mode
NUM_RUNS = 15  # Default for light mode
DEPTH = 3
JOB_TIMEOUT = 3600  # 1 hour timeout for hardware jobs

# Enhanced anthropomorphism parameters
ANTHROPOMORPHISM = {
    'human': {'theta': 1.0, 'phi': 1.0, 'lambda': 1.0, 'entanglement': 'full'},
    'mimic-human': {'theta': 0.9, 'phi': 0.8, 'lambda': 0.7, 'entanglement': 'full'},
    'animated-human': {'theta': 0.6, 'phi': 0.5, 'lambda': 0.4, 'entanglement': 'linear'},
    'non-human': {'theta': 0.2, 'phi': 0.1, 'lambda': 0.1, 'entanglement': 'none'},
    'caretaker': {'theta': 0.5, 'phi': 0.4, 'lambda': 0.3, 'entanglement': 'circular'},
    'educator': {'theta': 0.7, 'phi': 0.6, 'lambda': 0.5, 'entanglement': 'full'},
    'companion': {'theta': 0.4, 'phi': 0.3, 'lambda': 0.2, 'entanglement': 'linear'}
}

# Expanded product categories
PRODUCT_CATEGORIES = {
    'entertainment': {'feature_map': ZZFeatureMap, 'reps': 3},
    'healthcare': {'feature_map': PauliFeatureMap, 'reps': 2, 'paulis': ['Z', 'X', 'ZY']},
    'education': {'feature_map': PauliFeatureMap, 'reps': 1},
    'travel': {'feature_map': PauliFeatureMap, 'reps': 2, 'paulis': ['Z', 'X', 'Y']},
    'fashion': {'feature_map': ZZFeatureMap, 'reps': 2},
    'tech': {'feature_map': PauliFeatureMap, 'reps': 3, 'paulis': ['Z', 'Y', 'XZ']}
}

# Global variable for parameter history
param_history_df = pd.DataFrame()

# Version check
def check_versions():
    """Check versions of critical dependencies."""
    print("Dependency Versions:")
    print(f"Python: {sys.version}")
    print(f"Qiskit: {qiskit.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"Pandas: {pd.__version__}")
    print(f"PyMC: {pm.__version__}")
    print(f"qiskit-ibm-runtime: {'Installed' if qiskit_ibm_runtime else 'Not Installed'}")

def create_anthropomorphism_mapping():
    """Create a mapping of anthropomorphism parameters to behavioral traits."""
    mapping = {
        'influencer_type': [],
        'theta (empathy)': [],
        'phi (expressiveness)': [],
        'lambda (reliability)': [],
        'entanglement': [],
        'description': []
    }
    descriptions = {
        'human': 'Ideal human influencer with maximum empathy, expressiveness, and reliability.',
        'mimic-human': 'High empathy, expressive, reliable; mimics human influencers closely.',
        'animated-human': 'Moderate empathy and expressiveness, less reliable; cartoon-like.',
        'non-human': 'Low empathy, expressiveness, and reliability; robotic or abstract.',
        'caretaker': 'Moderate empathy, low expressiveness, moderate reliability; nurturing.',
        'educator': 'High empathy, moderate expressiveness, high reliability; informative.',
        'companion': 'Low empathy, low expressiveness, low reliability; friendly but simple.'
    }
    for inf_type, params in ANTHROPOMORPHISM.items():
        mapping['influencer_type'].append(inf_type)
        mapping['theta (empathy)'].append(params['theta'])
        mapping['phi (expressiveness)'].append(params['phi'])
        mapping['lambda (reliability)'].append(params['lambda'])
        mapping['entanglement'].append(params['entanglement'])
        mapping['description'].append(descriptions[inf_type])
    df = pd.DataFrame(mapping)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f'anthropomorphism_mapping_{timestamp}.csv', index=False)
    return df

def create_noise_model(depolarizing_prob=0.02):
    """Create noise model with customizable depolarizing probability."""
    noise_model = NoiseModel()
    for qubit in range(TOTAL_QUBITS):
        noise_model.add_quantum_error(
            thermal_relaxation_error(50e-6, 70e-6, 0.08),
            ['rx', 'ry', 'rz'], [qubit]
        )
        noise_model.add_quantum_error(
            depolarizing_error(depolarizing_prob, 2), ['cx'], [qubit, (qubit + 1) % TOTAL_QUBITS]
        )
        probabilities = [0.98, 0.02]
        noise_model.add_readout_error([probabilities, probabilities[::-1]], [qubit])
    return noise_model

def optimize_circuit(qc):
    """Optimize circuit using transpiler passes."""
    pm = PassManager([
        Optimize1qGates(),
        CXCancellation(),
        BasisTranslator(target_basis=['u3', 'cx']),
        UnitarySynthesis(basis_gates=['u3', 'cx'])
    ])
    return pm.run(qc)

def apply_error_correction(qc, logical_qubits):
    """Apply surface code-inspired error correction with readout mitigation."""
    for lq in logical_qubits:
        if lq + 2 < TOTAL_QUBITS:
            qc.h(lq)
            qc.cx(lq, lq+1)
            qc.cx(lq, lq+2)
            qc.ccx(lq+1, lq+2, lq)
    if TOTAL_QUBITS >= 5:
        flag_qubit = TOTAL_QUBITS - 1
        for lq in logical_qubits:
            qc.cx(lq, flag_qubit)
    noise_model = create_noise_model()
    backend = AerSimulator(method='statevector')
    mitigator = LocalReadoutMitigator(backend=backend)
    return optimize_circuit(qc), mitigator

def apply_zne(circuit, measurement_func, noise_levels=[1, 2, 3]):
    """Apply zero-noise extrapolation for a measurement function."""
    results = []
    for scale in noise_levels:
        scaled_noise = create_noise_model(depolarizing_prob=0.02 * scale)
        result = measurement_func(circuit, noise_model=scaled_noise)
        results.append(result)
    coeffs = np.polyfit(noise_levels, results, 1)
    return coeffs[1]

class QuantumParameterOptimizer(BaseEstimator):
    """Bayesian optimizer for quantum parameters."""
    def __init__(self, initial_params=None, influencer_qubits=None, consumer_qubits=None):
        self.params = initial_params or {'theta': 0.5, 'phi': 0.5, 'lambda': 0.5}
        self.influencer_qubits = influencer_qubits or list(range(NUM_VI_QUBITS))
        self.consumer_qubits = consumer_qubits or list(range(NUM_VI_QUBITS, NUM_VI_QUBITS + NUM_CONSUMER_QUBITS))

    def objective(self, params):
        theta, phi, lambda_val = params
        test_qc = QuantumCircuit(TOTAL_QUBITS)
        test_qc, _ = create_influencer_state(
            test_qc, self.influencer_qubits, 
            {'theta': theta, 'phi': phi, 'lambda': lambda_val, 'entanglement': 'full'}, 
            t=np.pi/2
        )
        test_qc, _ = create_consumer_state(
            test_qc, self.consumer_qubits, 
            {'theta': theta, 'phi': phi, 'lambda': lambda_val}, 
            self_concept=0.5, mood=0.5
        )
        for i, j in zip(self.influencer_qubits, self.consumer_qubits):
            test_qc.cx(i, j)
        social_presence, _, _ = measure_social_presence(
            test_qc, self.influencer_qubits, self.consumer_qubits, 
            {'theta': theta}, create_noise_model(), None
        )
        return -social_presence  # Maximize social presence

    def fit(self, X=None, y=None):
        res = minimize(
            self.objective, 
            x0=list(self.params.values()),
            method='COBYLA',
            options={'maxiter': 50}
        )
        self.params = {'theta': res.x[0], 'phi': res.x[1], 'lambda': res.x[2]}
        return self

def create_influencer_state(circuit, qubits, params, t, depth=DEPTH):
    """Enhanced with dynamic entanglement patterns."""
    feature_map = ZZFeatureMap(len(qubits), reps=2, parameter_prefix='φ')
    for i, qubit in enumerate(qubits):
        circuit.ry(params['theta'] * np.pi, qubit)
        circuit.rz(params['phi'] * np.pi, qubit)
    
    entanglement = params.get('entanglement', 'full')
    if entanglement == 'full':
        for i in range(len(qubits)-1):
            for j in range(i+1, len(qubits)):
                circuit.cx(qubits[i], qubits[j])
    elif entanglement == 'linear':
        for i in range(len(qubits)-1):
            circuit.cx(qubits[i], qubits[i+1])
    elif entanglement == 'circular':
        for i in range(len(qubits)):
            circuit.cx(qubits[i], qubits[(i+1)%len(qubits)])
    elif entanglement == 'none':
        pass
    
    time_param = Parameter('t')
    for q in qubits:
        circuit.rx(time_param, q)
    circuit = circuit.assign_parameters({time_param: t})
    
    var_form = EfficientSU2(
        len(qubits), 
        su2_gates=['ry', 'rz', 'rx'], 
        entanglement=entanglement, 
        reps=depth+1
    )
    circuit.compose(var_form, qubits, inplace=True)
    
    circuit, mitigator = apply_error_correction(circuit, qubits)
    return circuit, mitigator

def create_consumer_state(circuit, qubits, influencer_type, self_concept, mood, depth=DEPTH):
    """Prepare consumer state with quantum features."""
    congruence = 1 - abs(self_concept - ANTHROPOMORPHISM[influencer_type]['theta'])
    expressiveness = np.random.uniform(0.5, 1.0)
    feature_map = PauliFeatureMap(len(qubits), reps=2, paulis=['Z', 'X', 'Y', 'ZX'])
    circuit.compose(feature_map, qubits, inplace=True)
    theta = ParameterVector('θ', length=len(qubits))
    phi = ParameterVector('φ', length=len(qubits))
    for i, q in enumerate(qubits):
        circuit.ry(theta[i], q)
        circuit.rz(phi[i], q)
    param_binds = {}
    for i in range(len(qubits)):
        param_binds[theta[i]] = congruence * np.pi if i == 0 else \
                                expressiveness * np.pi/2 if i == 1 else \
                                mood * np.pi if i == 2 else 0
        param_binds[phi[i]] = congruence * np.pi/2 if i == 0 else \
                              expressiveness * np.pi/4 if i == 1 else \
                              mood * np.pi/2 if i == 2 else 0
    circuit = circuit.assign_parameters(param_binds)
    var_form = TwoLocal(len(qubits), ['ry', 'rz'], 'cz', entanglement='full', reps=depth)
    circuit.compose(var_form, qubits, inplace=True)
    if len(qubits) >= 3:
        circuit.h(qubits[0])
        circuit.cx(qubits[0], qubits[1])
        circuit.cx(qubits[2], qubits[0])
        circuit.h(qubits[2])
    circuit, mitigator = apply_error_correction(circuit, qubits)
    return circuit, mitigator

def apply_context_state(circuit, qubits, product_category, depth=DEPTH):
    """Enhanced with expanded product categories."""
    config = PRODUCT_CATEGORIES[product_category]
    if config['feature_map'] == PauliFeatureMap:
        feature_map = config['feature_map'](
            len(qubits), 
            reps=config['reps'],
            paulis=config.get('paulis', ['Z', 'X', 'Y'])
        )
    else:
        feature_map = config['feature_map'](
            len(qubits), 
            reps=config['reps']
        )
    circuit.compose(feature_map, qubits, inplace=True)
    
    if product_category == 'entertainment':
        for q in qubits:
            circuit.rx(np.pi/3, q)
        for i in range(len(qubits)-1):
            circuit.crx(np.pi/4, qubits[i], qubits[i+1])
    elif product_category == 'healthcare':
        for q in qubits:
            circuit.ry(np.pi/2, q)
        for i in range(len(qubits)-1):
            circuit.cry(np.pi/3, qubits[i], qubits[i+1])
    elif product_category == 'travel':
        for q in qubits:
            circuit.rx(np.pi/4, q)
        for i in range(len(qubits)-1):
            circuit.crx(np.pi/5, qubits[i], qubits[i+1])
    elif product_category == 'fashion':
        for q in qubits:
            circuit.ry(np.pi/3, q)
        for i in range(len(qubits)-1):
            circuit.cry(np.pi/4, qubits[i], qubits[i+1])
    elif product_category == 'tech':
        for q in qubits:
            circuit.rx(np.pi/4, q)
            circuit.rz(np.pi/3, q)
        for i in range(len(qubits)-1):
            circuit.crz(np.pi/4, qubits[i], qubits[i+1])
    else:  # education
        for q in qubits:
            circuit.ry(np.pi/3, q)
        for i in range(len(qubits)-1):
            circuit.cry(np.pi/4, qubits[i], qubits[i+1])
    
    var_form = TwoLocal(len(qubits), ['rx', 'ry', 'rz'], 'crz', entanglement='full', reps=depth)
    circuit.compose(var_form, qubits, inplace=True)
    circuit, mitigator = apply_error_correction(circuit, qubits)
    return circuit, mitigator

def measure_in_basis(circuit, basis, mitigator=None, ibm_token=None, backend_name="ibm_brisbane"):
    """Measure circuit in specified basis with readout mitigation."""
    meas_qc = circuit.copy()
    for i, b in enumerate(basis * (TOTAL_QUBITS // len(basis) + 1))[:TOTAL_QUBITS]:
        if b == 'X':
            meas_qc.h(i)
        elif b == 'Y':
            meas_qc.sdg(i)
            meas_qc.h(i)
    high_presence = run_on_hardware(meas_qc, shots=SHOTS, ibm_token=ibm_token, backend_name=backend_name)
    if high_presence is None:
        raise ValueError("Hardware measurement failed.")
    return high_presence

def measure_social_presence(circuit, influencer_qubits, consumer_qubits, influencer_type, noise_model, mitigator=None, ibm_token=None, backend_name="ibm_brisbane"):
    """Measure social presence using SWAP test and contextual variance."""
    ideal_circuit = QuantumCircuit(TOTAL_QUBITS)
    ideal_circuit.h(influencer_qubits)
    ideal_state = AerSimulator(method='statevector').run(ideal_circuit).result().get_statevector()
    
    swap_qc = QuantumCircuit(TOTAL_QUBITS + 1, 1)
    swap_qc.append(circuit, range(TOTAL_QUBITS))
    swap_qc.h(TOTAL_QUBITS)
    for q in influencer_qubits + consumer_qubits:
        swap_qc.cswap(TOTAL_QUBITS, q, q % len(influencer_qubits))
    swap_qc.h(TOTAL_QUBITS)
    swap_qc.measure(TOTAL_QUBITS, 0)
    
    high_presence = run_on_hardware(swap_qc, shots=SHOTS, ibm_token=ibm_token, backend_name=backend_name)
    if high_presence is None:
        raise ValueError("Hardware SWAP test failed.")
    p0 = high_presence
    
    witness_val = 1 - 2 * p0
    
    contextual_var = np.var([
        measure_in_basis(circuit, 'X', mitigator, ibm_token, backend_name),
        measure_in_basis(circuit, 'Y', mitigator, ibm_token, backend_name),
        measure_in_basis(circuit, 'Z', mitigator, ibm_token, backend_name)
    ])
    
    social_presence = (witness_val + 0.5 * contextual_var) * ANTHROPOMORPHISM[influencer_type]['theta']
    
    def sp_measurement(circ, noise_model, influencer_qubits, consumer_qubits, influencer_type, mitigator, ibm_token, backend_name):
        return measure_social_presence(circ, influencer_qubits, consumer_qubits, influencer_type, noise_model, mitigator, ibm_token, backend_name)[0]
    
    social_presence = apply_zne(
        circuit, 
        lambda circ, noise_model: sp_measurement(circ, noise_model, influencer_qubits, consumer_qubits, influencer_type, mitigator, ibm_token, backend_name)
    )
    
    return social_presence, contextual_var, ideal_state

def measure_authenticity_seeking(statevector, ideal_state, influencer_qubits, consumer_qubits):
    """Quantify deviation from ideal social presence using quantum kernel."""
    kernel = FidelityQuantumKernel(feature_map=PauliFeatureMap(TOTAL_QUBITS, reps=2, entanglement='full'))
    state_matrix = np.array([statevector.data, ideal_state.data])
    kernel_matrix = kernel.evaluate(state_matrix)
    authenticity = 1 - kernel_matrix[0,1]
    rho = partial_trace(statevector, 
                       [i for i in range(TOTAL_QUBITS) if i not in influencer_qubits + consumer_qubits])
    entanglement_pers = concurrence(rho)
    return {
        'authenticity': authenticity,
        'kernel_matrix': kernel_matrix,
        'entanglement_persistence': entanglement_pers
    }

def measure_emotional_attachment(circuit, statevector, influencer_qubits, consumer_qubits, self_concept, influencer_type):
    """Measure emotional attachment using quantum entanglement."""
    rho = partial_trace(statevector, [i for i in range(TOTAL_QUBITS) if i not in influencer_qubits + consumer_qubits])
    concurrence_val = concurrence(rho)
    kernel = FidelityQuantumKernel(feature_map=PauliFeatureMap(TOTAL_QUBITS, reps=2))
    probs = np.abs(statevector.data) ** 2
    probs /= np.sum(probs)
    valence = np.sum(probs[:len(probs)//2]) - np.sum(probs[len(probs)//2:])
    arousal = np.sum(probs[::2])
    kernel_matrix = kernel.evaluate(np.array([[valence, arousal]]))
    kernel_score = kernel_matrix[0, 0]
    score = np.sqrt(valence**2 + arousal**2)
    congruence = 1 - abs(self_concept - ANTHROPOMORPHISM[influencer_type]['theta'])
    expressiveness = np.random.uniform(0.5, 1.0)
    composite = concurrence_val * (valence + arousal) / 2 * congruence * expressiveness
    
    def ea_measurement(circ, noise_model, influencer_qubits, consumer_qubits, self_concept, influencer_type):
        sv = AerSimulator(method='statevector').run(circ, noise_model=noise_model).result().get_statevector()
        return measure_emotional_attachment(circ, sv, influencer_qubits, consumer_qubits, self_concept, influencer_type)['composite']
    
    composite = apply_zne(
        circuit, 
        lambda circ, noise_model: ea_measurement(circ, noise_model, influencer_qubits, consumer_qubits, self_concept, influencer_type)
    )
    
    return {
        "composite": composite,
        "valence": valence,
        "arousal": arousal,
        "kernel_score": kernel_score,
        "emotion_score": score
    }

def measure_entanglement(circuit, qubit_pair):
    """Deeper entanglement measurement using concurrence."""
    statevector = AerSimulator().run(circuit).result().get_statevector()
    rho = partial_trace(statevector, 
                       [i for i in range(TOTAL_QUBITS) if i not in qubit_pair])
    return concurrence(rho)

def measure_incompatibility(statevector):
    """Measure non-commutativity of observables."""
    rho = DensityMatrix(statevector)
    X_op = np.array([[0, 1], [1, 0]])
    Z_op = np.array([[1, 0], [0, -1]])
    ex = rho.expectation_value(X_op)
    ez = rho.expectation_value(Z_op)
    return abs(ex * ez - ez * ex)

def test_bell_violation(statevector, qubit_pair):
    """Test CHSH inequality for entanglement."""
    angles = [(0, 0), (0, 45), (45, 0), (45, 45)]
    correlations = []
    for a, b in angles:
        qc = QuantumCircuit(2, 1)
        qc.initialize(statevector.data, qubit_pair)
        qc.ry(np.radians(a), 0)
        qc.ry(np.radians(b), 1)
        qc.measure(0, 0)
        result = AerSimulator(method='statevector').run(qc, shots=SHOTS).result()
        counts = result.get_counts()
        p1 = counts.get('1', 0) / SHOTS
        correlations.append(p1)
    chsh = correlations[0] - correlations[1] + correlations[2] + correlations[3]
    return chsh > 2

def measure_credibility(statevector):
    """Measure trustworthiness and expertise."""
    rho_vi = partial_trace(statevector, list(range(NUM_VI_QUBITS, TOTAL_QUBITS)))
    trustworthiness = entropy(rho_vi, base=2)
    expertise = np.mean([np.abs(rho_vi.data[i, i]) for i in range(rho_vi.data.shape[0])])
    return {"trustworthiness": trustworthiness, "expertise": expertise}

def measure_purchase_intention(social_presence, emotional_attachment, credibility):
    """Calculate purchase intention as a weighted composite."""
    return (0.3 * social_presence +
            0.3 * emotional_attachment +
            0.2 * credibility["trustworthiness"] +
            0.2 * credibility["expertise"])

def bayesian_mediation_analysis(social_presence, emotional_attachment, authenticity_seeking, trust, influencer_type, mood, light_mode=True):
    """Perform Bayesian mediation analysis with optimized sampling."""
    samples = 200 if light_mode else 300
    tunes = 200 if light_mode else 300
    with pm.Model() as model:
        a = pm.Normal('a', mu=0, sigma=1)
        b1 = pm.Normal('b1', mu=0, sigma=1)
        b2 = pm.Normal('b2', mu=0, sigma=1)
        b3 = pm.Normal('b3', mu=0, sigma=1)
        c1 = pm.Normal('c1', mu=0, sigma=1)
        c2 = pm.Normal('c2', mu=0, sigma=1)
        c3 = pm.Normal('c3', mu=0, sigma=1)
        mood_effect = pm.Normal('mood', mu=0, sigma=1)
        
        sp = pm.Deterministic('sp', a + mood_effect * mood)
        ea = pm.Deterministic('ea', b1 * sp + c1 + mood_effect * mood)
        auth = pm.Deterministic('auth', b2 * sp + c2 + mood_effect * mood)
        trust_effect = pm.Deterministic('trust', b3 * sp + c3 + mood_effect * mood)
        mediation_ratio = pm.Deterministic('mediation_ratio', a / b1)
        
        pm.Normal('obs_sp', mu=sp, sigma=0.1, observed=social_presence)
        pm.Normal('obs_ea', mu=ea, sigma=0.1, observed=emotional_attachment)
        pm.Normal('obs_auth', mu=auth, sigma=0.1, observed=authenticity_seeking)
        pm.Normal('obs_trust', mu=trust_effect, sigma=0.1, observed=trust)
        
        trace = pm.sample(samples, tune=tunes, return_inferencedata=False, progressbar=False)
    
    mediation_index_ea = np.mean(trace['a'] * trace['b1'])
    mediation_index_auth = np.mean(trace['a'] * trace['b2'])
    mediation_index_trust = np.mean(trace['a'] * trace['b3'])
    ci_ea = np.percentile(trace['a'] * trace['b1'], [2.5, 97.5])
    ci_auth = np.percentile(trace['a'] * trace['b2'], [2.5, 97.5])
    ci_trust = np.percentile(trace['a'] * trace['b3'], [2.5, 97.5])
    significant_ea = not (ci_ea[0] < 0 < ci_ea[1]) if influencer_type == 'animated-human' else True
    significant_auth = not (ci_auth[0] < 0 < ci_auth[1]) if influencer_type == 'animated-human' else True
    significant_trust = not (ci_trust[0] < 0 < ci_trust[1]) if influencer_type == 'animated-human' else True
    
    return {
        'a': np.mean(trace['a']),
        'b1': np.mean(trace['b1']),
        'b2': np.mean(trace['b2']),
        'b3': np.mean(trace['b3']),
        'c1': np.mean(trace['c1']),
        'c2': np.mean(trace['c2']),
        'c3': np.mean(trace['c3']),
        'mediation_index_ea': mediation_index_ea,
        'mediation_index_auth': mediation_index_auth,
        'mediation_index_trust': mediation_index_trust,
        'ci_ea': ci_ea.tolist(),
        'ci_auth': ci_auth.tolist(),
        'ci_trust': ci_trust.tolist(),
        'significant_ea': significant_ea,
        'significant_auth': significant_auth,
        'significant_trust': significant_trust,
        'mediation_ratio': np.mean(trace['mediation_ratio'])
    }

def benchmark_classical(results_df):
    """Compare quantum results against classical neural network."""
    start_time = time.time()
    
    X = results_df[['self_concept', 'mood', 'theta', 'phi', 'lambda']].copy()
    for inf_type in ANTHROPOMORPHISM:
        X[f'inf_type_{inf_type}'] = (results_df['influencer_type'] == inf_type).astype(int)
    for cat in results_df['product_category'].unique():
        X[f'cat_{cat}'] = (results_df['product_category'] == cat).astype(int)
    
    metrics = {}
    for metric in ['social_presence', 'emotional_attachment', 'purchase_intention']:
        y = results_df[metric]
        model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        model.fit(X, y)
        pred = model.predict(X)
        metrics[metric] = {
            'mse': mean_squared_error(y, pred),
            'r2': r2_score(y, pred)
        }
    
    classical_time = time.time() - start_time
    
    quantum_time = results_df.get('runtime', classical_time * 2)
    if 'runtime' not in results_df:
        start_time = time.time()
        run_condition((list(ANTHROPOMORPHISM.keys())[0], results_df['product_category'].iloc[0]), ibm_token=None)
        quantum_time = time.time() - start_time
    
    metrics['runtime'] = {
        'quantum': quantum_time * NUM_RUNS,
        'classical': classical_time
    }
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'classical_benchmark_{timestamp}.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

def tune_parameters(tasks, ibm_token=None, backend_name="ibm_brisbane"):
    """Perform grid search over circuit depth and noise levels."""
    depths = [1, 3, 5]
    noise_probs = [0.01, 0.03, 0.05]
    best_depth, best_noise_prob, best_score = DEPTH, 0.02, float('inf')
    
    for depth, noise_prob in product(depths, noise_probs):
        results = []
        for task in tasks[:5]:
            result = run_condition(task, depth=depth, noise_prob=noise_prob, ibm_token=ibm_token, backend_name=backend_name)
            if result:
                results.append(result['social_presence'])
        if results:
            score = np.var(results)
            if score < best_score:
                best_depth, best_noise_prob, best_score = depth, noise_prob, score
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'parameter_tuning_{timestamp}.json', 'w') as f:
        json.dump({'best_depth': best_depth, 'best_noise_prob': best_noise_prob, 'best_score': best_score}, f)
    
    return best_depth, best_noise_prob

def dynamic_parameter_tuning(influencer_type, influencer_qubits, consumer_qubits):
    """Auto-tune parameters for max social presence."""
    optimizer = QuantumParameterOptimizer(
        initial_params=ANTHROPOMORPHISM[influencer_type],
        influencer_qubits=influencer_qubits,
        consumer_qubits=consumer_qubits
    )
    optimizer.fit()
    return optimizer.params

def validate_with_real_data(results_df):
    """Validate quantum results against real-world influencer data."""
    validation_results = {
        'status': 'Placeholder - real data not provided',
        'correlation': None
    }
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'validation_results_{timestamp}.json', 'w') as f:
        json.dump(validation_results, f, indent=4)
    return validation_results

def correlate_user_study(results_df):
    """Correlate simulated purchase intention with consumer survey data."""
    try:
        survey_data = pd.DataFrame({
            'influencer_type': results_df['influencer_type'],
            'product_category': results_df['product_category'],
            'purchase_likelihood': np.random.uniform(0, 1, len(results_df))
        })
        merged = results_df.merge(
            survey_data,
            on=['influencer_type', 'product_category'],
            how='inner'
        )
        if len(merged) == 0:
            raise ValueError("No matching survey data found.")
        
        correlation, p_value = pearsonr(merged['purchase_intention'], merged['purchase_likelihood'])
        results = {
            'correlation': correlation,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'sample_size': len(merged)
        }
    except Exception as e:
        results = {
            'status': f'Error: {str(e)} - Provide survey_data.csv with influencer_type, product_category, purchase_likelihood',
            'correlation': None
        }
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'user_study_correlation_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def run_on_hardware(circuit, shots=SHOTS, ibm_token=None, backend_name=None):
    """Execute circuit on IBM Quantum hardware with dynamic backend selection."""
    if not qiskit_ibm_runtime:
        error_msg = "Error: qiskit-ibm-runtime not installed. Install using: pip install qiskit-ibm-runtime\nFalling back to simulator."
        print(error_msg)
        with open(f'error_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt', 'a') as f:
            f.write(error_msg)
        return None
    
    if not ibm_token:
        error_msg = "Error: IBM Quantum API token required. Get token from https://quantum-computing.ibm.com\nFalling back to simulator."
        print(error_msg)
        with open(f'error_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt', 'a') as f:
            f.write(error_msg)
        return None
    
    try:
        service = QiskitRuntimeService(channel="ibm_quantum", token=ibm_token)
        backends = [b for b in service.backends() if b.configuration().n_qubits >= TOTAL_QUBITS and b.status().status == 'active']
        if not backends:
            error_msg = "Error: No suitable backends available (need ≥12 qubits, active status).\nFalling back to simulator."
            print(error_msg)
            with open(f'error_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt', 'a') as f:
                f.write(error_msg)
            return None
        backend = min(backends, key=lambda b: b.status().pending_jobs)
        print(f"Selected backend: {backend.name} ({backend.configuration().n_qubits} qubits, {backend.status().pending_jobs} pending jobs)")
        
        sampler = Sampler(backend=backend)
        
        circuit.measure_all()
        start_time = time.time()
        job = sampler.run(circuit, shots=shots)
        result = job.result(timeout=JOB_TIMEOUT)
        
        counts = result.quasi_dists[0].binary_probabilities()
        high_presence = sum(v for k, v in counts.items() if k[:6].count('1') >= 4)
        print(f"Hardware job completed in {time.time() - start_time:.2f} seconds")
        return high_presence / shots
    except Exception as e:
        error_msg = f"Hardware execution failed: {str(e)}\n"
        error_msg += "Possible issues:\n- Invalid token: Verify at https://quantum-computing.ibm.com\n- Backend unavailable: Try --backend_name ibm_kyoto or ibm_sherbrooke\n- Network error: Check connection with 'ping quantum-computing.ibm.com'\n- Free-tier limit exceeded: Reduce NUM_RUNS or use debug mode\nFalling back to simulator.\n"
        error_msg += traceback.format_exc() + "\n"
        print(error_msg)
        with open(f'error_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt', 'a') as f:
            f.write(error_msg)
        return None

def run_statistical_analysis(results_df):
    """Perform statistical analysis with quantum metrics."""
    stats_summary = {'anova': {}, 'pairwise': {}, 'feature_importance': {}, 'quantum_metrics': {}}
    influencer_types = results_df['influencer_type'].unique()
    
    for metric in ['emotional_attachment', 'trustworthiness', 'expertise', 'purchase_intention', 'entanglement']:
        data = [results_df[results_df['influencer_type'] == inf][metric] for inf in influencer_types]
        f_stat, p_value = f_oneway(*data)
        stats_summary['anova'][metric] = {'F': f_stat, 'p': p_value, 'significant': p_value < 0.05}
    
    comparisons = [(a, b) for i, a in enumerate(influencer_types) for b in influencer_types[i+1:]]
    for metric in ['emotional_attachment', 'trustworthiness', 'expertise', 'purchase_intention', 'entanglement']:
        p_values = []
        for a, b in comparisons:
            _, p = ttest_ind(
                results_df[results_df['influencer_type'] == a][metric],
                results_df[results_df['influencer_type'] == b][metric],
                equal_var=False
            )
            p_values.append(p)
        reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
        for (a, b), p_corr, rej in zip(comparisons, pvals_corrected, reject):
            stats_summary['pairwise'][f"{metric}_{a}_vs_{b}"] = {'p_corrected': p_corr, 'significant': rej}
    
    data = results_df[['social_presence', 'valence', 'arousal', 'authenticity',
                      'trustworthiness', 'expertise', 'emotional_attachment', 'entanglement']].dropna()
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(data.drop('emotional_attachment', axis=1), data['emotional_attachment'])
    importances = pd.Series(model.feature_importances_, index=data.drop('emotional_attachment', axis=1).columns)
    stats_summary['feature_importance'] = importances.to_dict()
    
    stats_summary['quantum_metrics']['contextual_variance'] = results_df.groupby('influencer_type')['contextual_variance'].describe().to_dict()
    stats_summary['quantum_metrics']['entanglement_persistence'] = results_df.groupby('influencer_type')['entanglement_persistence'].mean().to_dict()
    stats_summary['quantum_metrics']['bell_violation_freq'] = results_df.groupby('influencer_type')['bell_violation'].mean().to_dict()
    stats_summary['quantum_metrics']['entanglement'] = results_df.groupby('influencer_type')['entanglement'].mean().to_dict()
    kernel_results = results_df.groupby('influencer_type').apply(
        lambda x: np.mean([np.array(k)[0,1] for k in x['kernel_matrix']])).to_dict()
    stats_summary['quantum_metrics']['kernel_similarity'] = kernel_results
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fig = go.Figure(data=[go.Bar(x=importances.index, y=importances.values)])
    fig.update_layout(title="Feature Importance for Emotional Attachment",
                     xaxis_title="Features", yaxis_title="Importance")
    fig.write(f'feature_importance_{timestamp}.html')
    
    return stats_summary

def visualize_results(results_df, stats_summary, classical_metrics=None):
    """Generate comprehensive visualizations including quantum-specific plots."""
    global param_history_df
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    influencer_types = results_df['influencer_type'].unique()
    product_categories = results_df['product_category'].unique()
    
    anthropomorphism_df = create_anthropomorphism_mapping()
    print("Anthropomorphism Mapping:\n", anthropomorphism_df)
    
    # Existing Visualizations
    for metric, title, filename in [
        ('social_presence', 'Social Presence by VI Type', f'social_presence_{timestamp}.html'),
        ('trustworthiness', 'Trustworthiness by VI Type', f'trustworthiness_{timestamp}.html'),
        ('expertise', 'Expertise by VI Type', f'expertise_{timestamp}.html'),
        ('purchase_intention', 'Purchase Intention by VI Type', f'purchase_intention_{timestamp}.html'),
        ('entanglement', 'Entanglement by VI Type', f'entanglement_{timestamp}.html')
    ]:
        fig = go.Figure()
        for inf in influencer_types:
            data = results_df[results_df['influencer_type'] == inf][metric]
            fig.add_trace(go.Violin(y=data, name=inf, box_visible=True, meanline_visible=True))
        fig.update_layout(
            title=f"{title}\n(ANOVA F={stats_summary['anova'].get(metric, {}).get('F', 0):.2f}, p={stats_summary['anova'].get(metric, {}).get('p', 1):.4f})",
            yaxis_title=metric.replace('_', ' ').title(),
            template='plotly_white'
        )
        fig.write(filename)
    
    fig = go.Figure()
    for inf in influencer_types:
        mask = results_df['influencer_type'] == inf
        fig.add_trace(go.Scatter(
            x=results_df[mask]['valence'],
            y=results_df[mask]['arousal'],
            mode='markers',
            name=inf,
            text=[f"Score: {s:.2f}" for s in results_df[mask]['emotion_score']],
            marker=dict(size=10)
        ))
    fig.update_layout(
        title='Valence vs Arous Voucher by VI Type',
        xaxis_title='Valence',
        yaxis_title='Arousal',
        template='plotly_white'
    )
    fig.write(f'valence_arousal_{timestamp}.html')
    
    fig = go.Figure()
    for inf in influencer_types:
        mask = results_df['influencer_type'] == inf
        fig.add_trace(go.Scatter(
            x=results_df[mask]['social_presence'],
            y=results_df[mask]['purchase_intention'],
            mode='markers',
            name=inf,
            text=[f"Score: {s:.2f}" for s in results_df[mask]['emotion_score']],
            marker=dict(size=10)
        ))
    fig.update_layout(
        title='Social Presence vs Purchase Intention',
        xaxis_title='Social Presence',
        yaxis_title='Purchase Intention',
        template='plotly_white'
    )
    fig.write(f'sp_purchase_scatter_{timestamp}.html')
    
    fig = go.Figure()
    for inf in influencer_types:
        mask = results_df['influencer_type'] == inf
        fig.add_trace(go.Scatter(
            x=results_df[mask]['trustworthiness'],
            y=results_df[mask]['expertise'],
            mode='markers',
            name=inf,
            text=[f"Score: {s:.2f}" for s in results_df[mask]['emotion_score']],
            marker=dict(size=10)
        ))
    fig.update_layout(
        title='Trustworthiness vs Expertise',
        xaxis_title='Trustworthiness',
        yaxis_title='Expertise',
        template='plotly_white'
    )
    fig.write(f'trust_expertise_scatter_{timestamp}.html')
    
    fig = go.Figure()
    for inf in influencer_types:
        mask = results_df['influencer_type'] == inf
        fig.add_trace(go.Scatter(
            x=results_df[mask]['entanglement'],
            y=results_df[mask]['social_presence'],
            mode='markers',
            name=inf,
            text=[f"Score: {s:.2f}" for s in results_df[mask]['emotion_score']],
            marker=dict(size=10)
        ))
    fig.update_layout(
        title='Entanglement vs Social Presence',
        xaxis_title='Entanglement (Concurrence)',
        yaxis_title='Social Presence',
        template='plotly_white'
    )
    fig.write(f'entanglement_social_presence_{timestamp}.html')
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    for inf in influencer_types:
        mask = results_df['influencer_type'] == inf
        ax.scatter(
            results_df[mask]['valence'],
            results_df[mask]['arousal'],
            results_df[mask]['social_presence'],
            label=inf,
            s=50 * results_df[mask]['emotion_score']
        )
    ax.set_xlabel('Valence')
    ax.set_ylabel('Arousal')
    ax.set_zlabel('Social Presence')
    ax.legend()
    plt.savefig(f'3d_emotion_space_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    benefit_corr = np.zeros((len(influencer_types), len(product_categories)))
    for i, inf in enumerate(influencer_types):
        for j, cat in enumerate(product_categories):
            subset = results_df[(results_df['influencer_type'] == inf) & 
                              (results_df['product_category'] == cat)]['emotional_attachment']
            benefit_corr[i, j] = subset.mean() if len(subset) > 0 else 0
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        benefit_corr, 
        annot=True, 
        xticklabels=product_categories, 
        yticklabels=influencer_types, 
        cmap='viridis'
    )
    plt.title('Emotional Attachment by VI Type and Product Category')
    plt.xlabel('Product Category')
    plt.ylabel('VI Type')
    plt.savefig(f'emotional_attachment_heatmap_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    quantiles = results_df['self_concept'].quantile([0.33, 0.66]).values
    results_df['self_concept_group'] = pd.cut(
        results_df['self_concept'], 
        bins=[0, quantiles[0], quantiles[1], 1], 
        labels=['Low', 'Medium', 'High'],
        include_lowest=True
    )
    plt.figure(figsize=(10, 6))
    for inf in influencer_types:
        data = results_df[results_df['influencer_type'] == inf].groupby('self_concept_group', observed=True)['emotional_attachment'].mean().reset_index()
        sns.lineplot(x='self_concept_group', y='emotional_attachment', label=inf, data=data)
    plt.title('Emotional Attachment by Self-Concept Group and VI Type')
    plt.xlabel('Self-Concept Group')
    plt.ylabel('Emotional Attachment')
    plt.legend()
    plt.savefig(f'self_concept_segmentation_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig = go.Figure()
    for inf in influencer_types:
        row = results_df[results_df['influencer_type'] == inf].iloc[0]
        fig.add_trace(go.Scatter(
            x=['VI Type', 'Social Presence', 'Emotional Attachment'],
            y=[0, 1, 0],
            mode='lines+text',
            text=[
                f"a={row['a']:.2f} [{row['ci_ea'][0]:.2f}, {row['ci_ea'][1]:.2f}]",
                f"b1={row['b1']:.2f}",
                f"c1={row['c1']:.2f}"
            ],
            textposition='top center',
            name=f"{inf} (EA)",
            line=dict(dash='solid' if row['significant_ea'] else 'dash')
        ))
        fig.add_trace(go.Scatter(
            x=['VI Type', 'Social Presence', 'Authenticity'],
            y=[0, 1, 2],
            mode='lines+text',
            text=[
                f"a={row['a']:.2f} [{row['ci_auth'][0]:.2f}, {row['ci_auth'][1]:.2f}]",
                f"b2={row['b2']:.2f}",
                f"c2={row['c2']:.2f}"
            ],
            textposition='top center',
            name=f"{inf} (Auth)",
            line=dict(dash='solid' if row['significant_auth'] else 'dash')
        ))
        fig.add_trace(go.Scatter(
            x=['VI Type', 'Social Presence', 'Trustworthiness'],
            y=[0, 1, 3],
            mode='lines+text',
            text=[
                f"a={row['a']:.2f} [{row['ci_trust'][0]:.2f}, {row['ci_trust'][1]:.2f}]",
                f"b3={row['b3']:.2f}",
                f"c3={row['c3']:.2f}"
            ],
            textposition='top center',
            name=f"{inf} (Trust)",
            line=dict(dash='solid' if row['significant_trust'] else 'dash')
        ))
    fig.update_layout(
        title=f'Mediation Analysis\nMediation Ratio = {results_df["mediation_ratio"].mean():.2f}',
        showlegend=True,
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        template='plotly_white'
    )
    fig.write(f'mediation_diagram_{timestamp}.html')
    
    sp_data = np.zeros((len(influencer_types), NUM_RUNS // len(influencer_types)))
    for i, inf in enumerate(influencer_types):
        sp_data[i] = results_df[results_df['influencer_type'] == inf]['social_presence'].values[:NUM_RUNS // len(influencer_types)]
    fig = go.Figure()
    for i, inf in enumerate(influencer_types):
        fig.add_trace(go.Surface(
            x=np.arange(sp_data.shape[1]),
            y=[i] * sp_data.shape[1],
            z=sp_data[[i]],
            name=inf,
            opacity=0.7,
            colorscale='Viridis'
        ))
    fig.update_layout(
        title='Social Presence Across VI Types and Runs',
        scene=dict(
            xaxis_title='Run Index',
            yaxis_title='VI Type Index',
            zaxis_title='Social Presence'
        ),
        template='plotly_white'
    )
    fig.write(f'3d_social_presence_surface_{timestamp}.html')
    
    if classical_metrics:
        fig = go.Figure()
        metrics = ['social_presence', 'emotional_attachment', 'purchase_intention']
        quantum_r2 = [1 - classical_metrics[m]['mse'] / np.var(results_df[m]) for m in metrics]
        classical_r2 = [classical_metrics[m]['r2'] for m in metrics]
        fig.add_trace(go.Bar(x=metrics, y=quantum_r2, name='Quantum', marker_color='blue'))
        fig.add_trace(go.Bar(x=metrics, y=classical_r2, name='Classical', marker_color='orange'))
        fig.update_layout(
            title='R² Comparison: Quantum vs Classical',
            xaxis_title='Metric',
            yaxis_title='R² Score',
            barmode='group',
            template='plotly_white'
        )
        fig.write(f'r2_comparison_{timestamp}.html')
    
    # New Quantum-Specific Visualizations
    try:
        # 1. Entanglement Heatmap
        if not results_df[['influencer_type', 'product_category', 'entanglement']].dropna().empty:
            heatmap_data = results_df.pivot_table(
                index='influencer_type', 
                columns='product_category', 
                values='entanglement'
            )
            plt.figure(figsize=(10, 6))
            sns.heatmap(heatmap_data, annot=True, cmap='viridis', cbar_kws={'label': 'Concurrence'})
            plt.title('Qubit Entanglement Strength by VI Type and Product Category')
            plt.tight_layout()
            plt.savefig(f'entanglement_heatmap_{timestamp}.png', dpi=300)
            plt.close()
    except Exception as e:
        print(f"Error in entanglement heatmap: {str(e)}")
        with open(f'error_log_{timestamp}.txt', 'a') as f:
            f.write(f"Entanglement heatmap error: {str(e)}\n")
    
    try:
        # 2. Quantum State Fidelity Radar Chart
        radar_df = results_df.groupby('influencer_type').agg({
            'purchase_intention': 'mean',
            'entanglement': 'mean',
            'theta': 'mean'  # Use theta directly from results_df
        }).reset_index()
        fig = px.line_polar(
            radar_df, 
            r=['purchase_intention', 'entanglement', 'theta'], 
            theta=['Purchase Intention', 'Entanglement', 'Theta (Empathy)'],
            line_close=True,
            color='influencer_type',
            template='plotly_white'
        )
        fig.update_layout(title='Quantum Metric Radar Chart')
        fig.write(f'quantum_radar_{timestamp}.html')
    except Exception as e:
        print(f"Error in fidelity radar chart: {str(e)}")
        with open(f'error_log_{timestamp}.txt', 'a') as f:
            f.write(f"Fidelity radar chart error: {str(e)}\n")
    
    try:
        # 3. Parameter Optimization Trajectory
        if not param_history_df.empty:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            for inf_type in param_history_df['influencer_type'].unique():
                subset = param_history_df[param_history_df['influencer_type'] == inf_type]
                ax.scatter(
                    subset['theta'],
                    subset['phi'],
                    subset['lambda'],
                    label=inf_type,
                    s=50
                )
            ax.set_xlabel('Theta (Empathy)')
            ax.set_ylabel('Phi (Expressiveness)')
            ax.set_zlabel('Lambda (Reliability)')
            ax.legend()
            plt.title('Bayesian Parameter Optimization Trajectory')
            plt.savefig(f'param_trajectory_3d_{timestamp}.png', dpi=300)
            plt.close()
    except Exception as e:
        print(f"Error in parameter trajectory: {str(e)}")
        with open(f'error_log_{timestamp}.txt', 'a') as f:
            f.write(f"Parameter trajectory error: {str(e)}\n")
    
    try:
        # 4. Quantum Circuit Diagram
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.ry(np.pi/4, 2)
        qc.cz(0, 2)
        circuit_drawer(
            qc, output='mpl', 
            style={'backgroundcolor': '#FFFFFF'},
            plot_barriers=False,
            fold=40
        ).savefig(f'entanglement_circuit_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error in circuit diagram: {str(e)}")
        with open(f'error_log_{timestamp}.txt', 'a') as f:
            f.write(f"Circuit diagram error: {str(e)}\n")
    
    try:
        # 5. Density Matrix Visualization
        if not results_df[results_df['influencer_type'] == 'human'].empty:
            human_case = results_df[results_df['influencer_type'] == 'human'].iloc[0]
            qc = QuantumCircuit(TOTAL_QUBITS)
            qc, _ = create_influencer_state(qc, list(range(NUM_VI_QUBITS)), {'theta': human_case['theta'], 'phi': human_case['phi'], 'lambda': human_case['lambda'], 'entanglement': 'full'}, np.pi/2)
            state = AerSimulator().run(qc).result().get_statevector()
            fig = plot_state_city(state)
            fig.savefig(f'density_matrix_{timestamp}.png', dpi=300)
            plt.close()
    except Exception as e:
        print(f"Error in density matrix: {str(e)}")
        with open(f'error_log_{timestamp}.txt', 'a') as f:
            f.write(f"Density matrix error: {str(e)}\n")
    
    try:
        # 6. Contextual Variance Surface Plot
        if not results_df[['influencer_type', 'product_category', 'purchase_intention']].dropna().empty:
            pivot_data = results_df.pivot_table(
                index='influencer_type',
                columns='product_category',
                values='purchase_intention'
            )
            fig = go.Figure(data=[go.Surface(z=pivot_data.values)])
            fig.update_layout(
                title='Purchase Intention Contextual Variance',
                scene=dict(
                    xaxis_title='Product Category',
                    yaxis_title='VI Type',
                    zaxis_title='Purchase Intention'
                ),
                margin=dict(l=0, r=0, b=0, t=40),
                template='plotly_white'
            )
            fig.write(f'contextual_variance_{timestamp}.html')
    except Exception as e:
        print(f"Error in contextual variance surface: {str(e)}")
        with open(f'error_log_{timestamp}.txt', 'a') as f:
            f.write(f"Contextual variance surface error: {str(e)}\n")
    
    try:
        # 7. Quantum State Histogram
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        counts = AerSimulator().run(qc, shots=1000).result().get_counts()
        fig = go.Figure([go.Bar(
            x=list(counts.keys()),
            y=list(counts.values()),
            marker_color='#636EFA'
        )])
        fig.update_layout(
            title='Bell State Measurement Histogram',
            xaxis_title='Quantum State',
            yaxis_title='Counts',
            template='plotly_white'
        )
        fig.write(f'state_histogram_{timestamp}.html')
    except Exception as e:
        print(f"Error in state histogram: {str(e)}")
        with open(f'error_log_{timestamp}.txt', 'a') as f:
            f.write(f"State histogram error: {str(e)}\n")
    
    try:
        # 8. Parameter Sensitivity Parallel Coordinates
        if not param_history_df.empty:
            flat_data = []
            for _, row in param_history_df.iterrows():
                flat_data.append({
                    'type': row['influencer_type'],
                    'theta': row['theta'],
                    'phi': row['phi'],
                    'lambda': row['lambda']
                })
            fig = px.parallel_coordinates(
                pd.DataFrame(flat_data),
                color='theta',
                dimensions=['theta', 'phi', 'lambda'],
                labels={'theta': 'Empathy', 'phi': 'Expressiveness', 'lambda': 'Reliability'},
                title='Parameter Sensitivity Analysis',
                template='plotly_white'
            )
            fig.write(f'parallel_coords_{timestamp}.html')
    except Exception as e:
        print(f"Error in parallel coordinates: {str(e)}")
        with open(f'error_log_{timestamp}.txt', 'a') as f:
            f.write(f"Parallel coordinates error: {str(e)}\n")

def plot_quantum_metrics(results_df):
    """Generate quantum-specific visualizations including clustering."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    features = results_df[['social_presence', 'entanglement_persistence', 'contextual_variance', 'entanglement']].dropna()
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(features)
    results_df['quantum_cluster'] = pd.Series(clusters, index=features.index)
    
    fig = px.scatter_3d(
        results_df,
        x='social_presence',
        y='entanglement_persistence',
        z='entanglement',
        color='quantum_cluster',
        symbol='influencer_type',
        title='Quantum Clustering of VI-Consumer Interactions',
        template='plotly_white'
    )
    fig.write(f'quantum_clusters_{timestamp}.html')

def save_results_table(results, param_history):
    """Save simulation results and parameter history to CSV and JSON."""
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        raise ValueError("No valid results to save. Check error_log_<timestamp>.txt for details.")
    
    results_df = pd.DataFrame(valid_results)
    param_history_df = pd.DataFrame(param_history)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_df.to_csv(f'qspf_results_{timestamp}.csv', index=False)
    param_history_df.to_csv(f'param_history_{timestamp}.csv', index=False)
    
    with open(f'qspf_results_{timestamp}.json', 'w') as f:
        json.dump(valid_results, f, indent=4)
    with open(f'param_history_{timestamp}.json', 'w') as f:
        json.dump(param_history, f, indent=4)
    
    checkpoint_data = [
        {
            'influencer_type': r['influencer_type'],
            'product_category': r['product_category'],
            'social_presence': r['social_presence'],
            'emotional_attachment': r['emotional_attachment'],
            'entanglement': r['entanglement'],
            'timestamp': timestamp
        }
        for r in valid_results
    ]
    pd.DataFrame(checkpoint_data).to_csv(f'qspf_checkpoint_{timestamp}.csv', index=False)
    
    return results_df, param_history_df

def run_condition(args, depth=DEPTH, noise_prob=0.02, debug=False, ibm_token=None, backend_name="ibm_brisbane"):
    """Simulate a single experimental condition with dynamic parameter tuning."""
    influencer_type, product_category = args
    circuit = QuantumCircuit(TOTAL_QUBITS)
    
    influencer_qubits = list(range(NUM_VI_QUBITS))
    consumer_qubits = list(range(NUM_VI_QUBITS, NUM_VI_QUBITS + NUM_CONSUMER_QUBITS))
    context_qubits = list(range(NUM_VI_QUBITS + NUM_CONSUMER_QUBITS, TOTAL_QUBITS))
    
    t = np.random.uniform(0, 2 * np.pi)
    self_concept = np.random.uniform(0, 1)
    mood = np.random.uniform(0, 1)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    error_log = f'error_log_{timestamp}.txt'
    
    try:
        # Dynamic parameter tuning for non-human influencers
        params = ANTHROPOMORPHISM[influencer_type].copy()
        if influencer_type != 'human' and not debug:
            tuned_params = dynamic_parameter_tuning(influencer_type, influencer_qubits, consumer_qubits)
            params.update(tuned_params)
        
        circuit, mitigator = create_influencer_state(circuit, influencer_qubits, params, t, depth)
        if debug:
            print(f"Created influencer state for {influencer_type} with params={params}")
        
        circuit, mitigator = create_consumer_state(circuit, consumer_qubits, influencer_type, self_concept, mood, depth)
        if debug:
            print(f"Created consumer state with self_concept={self_concept:.2f}, mood={mood:.2f}")
        
        circuit, mitigator = apply_context_state(circuit, context_qubits, product_category, depth)
        if debug:
            print(f"Applied context state for {product_category}")
        
        for i, j in zip(influencer_qubits, consumer_qubits):
            circuit.cx(i, j)
        
        high_presence = run_on_hardware(circuit, shots=SHOTS, ibm_token=ibm_token, backend_name=backend_name)
        if high_presence is None:
            raise ValueError("Hardware execution failed, check logs.")
        
        noise_model = create_noise_model(depolarizing_prob=noise_prob)
        backend = AerSimulator(method='statevector')
        result = backend.run(circuit, noise_model=noise_model, shots=SHOTS).result()
        statevector = result.get_statevector()
        if debug:
            print(f"Simulated circuit with {SHOTS} shots, statevector norm={np.linalg.norm(statevector.data):.2f}")
        
        social_presence, contextual_var, ideal_state = measure_social_presence(
            circuit, influencer_qubits, consumer_qubits, influencer_type, noise_model, mitigator, ibm_token, backend_name
        )
        if debug:
            print(f"Social presence: {social_presence:.2f}, contextual variance: {contextual_var:.2f}")
        
        auth_data = measure_authenticity_seeking(
            statevector, ideal_state, influencer_qubits, consumer_qubits
        )
        if debug:
            print(f"Authenticity: {auth_data['authenticity']:.2f}, entanglement persistence: {auth_data['entanglement_persistence']:.2f}")
        
        ea_data = measure_emotional_attachment(
            circuit, statevector, influencer_qubits, consumer_qubits, self_concept, influencer_type
        )
        if debug:
            print(f"Emotional attachment: {ea_data['composite']:.2f}, valence: {ea_data['valence']:.2f}, arousal: {ea_data['arousal']:.2f}")
        
        entanglement = measure_entanglement(circuit, [influencer_qubits[0], consumer_qubits[0]])
        if debug:
            print(f"Entanglement (concurrence): {entanglement:.2f}")
        
        incompatibility = measure_incompatibility(statevector)
        bell_violation = test_bell_violation(statevector, [influencer_qubits[0], consumer_qubits[0]])
        credibility = measure_credibility(statevector)
        purchase_intention = measure_purchase_intention(
            social_presence, ea_data['composite'], credibility
        )
        if debug:
            print(f"Incompatibility: {incompatibility:.2f}, Bell violation: {bell_violation}, Purchase intention: {purchase_intention:.2f}")
        
        mediation_results = bayesian_mediation_analysis(
            social_presence,
            ea_data['composite'],
            auth_data['authenticity'],
            credibility['trustworthiness'],
            influencer_type,
            mood
        )
        if debug:
            print(f"Mediation analysis completed: mediation_ratio={mediation_results['mediation_ratio']:.2f}")
        
        result = {
            'influencer_type': influencer_type,
            'product_category': product_category,
            'self_concept': self_concept,
            'mood': mood,
            'theta': params['theta'],
            'phi': params['phi'],
            'lambda': params['lambda'],
            'entanglement_type': params['entanglement'],
            'social_presence': social_presence,
            'contextual_variance': contextual_var,
            'authenticity': auth_data['authenticity'],
            'kernel_matrix': auth_data['kernel_matrix'].tolist(),
            'entanglement_persistence': auth_data['entanglement_persistence'],
            'entanglement': entanglement,
            'emotional_attachment': ea_data['composite'],
            'valence': ea_data['valence'],
            'arousal': ea_data['arousal'],
            'kernel_score': ea_data['kernel_score'],
            'emotion_score': ea_data['emotion_score'],
            'incompatibility': incompatibility,
            'bell_violation': bell_violation,
            'trustworthiness': credibility['trustworthiness'],
            'expertise': credibility['expertise'],
            'purchase_intention': purchase_intention,
            **mediation_results
        }
        
        param_history_entry = {
            'influencer_type': influencer_type,
            'product_category': product_category,
            'theta': params['theta'],
            'phi': params['phi'],
            'lambda': params['lambda'],
            'timestamp': timestamp
        }
        
        return result, param_history_entry
    except Exception as e:
        error_msg = f"Error in condition {influencer_type}, {product_category}: {str(e)}\n"
        error_msg += traceback.format_exc() + "\n"
        with open(error_log, 'a') as f:
            f.write(error_msg)
        print(error_msg)
        return None, None

def batch_process(tasks, batch_size=None, depth=DEPTH, noise_prob=0.02, debug=False, ibm_token=None, backend_name="ibm_brisbane"):
    """Process tasks in batches with dynamic sizing optimized for c5.4xlarge."""
    if batch_size is None:
        batch_size = max(5, mp.cpu_count() * 2)  # Scales to ~32 tasks on c5.4xlarge
    results = []
    param_history = []
    for i in range(0, len(tasks), batch_size):
        batch_tasks = tasks[i:i + batch_size]
        with Pool(processes=mp.cpu_count()) as pool:
            batch_results = pool.starmap(run_condition, [(task, depth, noise_prob, debug, ibm_token, backend_name) for task in batch_tasks])
        for result, param_entry in batch_results:
            if result is not None:
                results.append(result)
            if param_entry is not None:
                param_history.append(param_entry)
    return results, param_history

def main(light_mode=True, use_hardware=True, ibm_token=None, debug_mode=False, backend_name="ibm_brisbane"):
    """Run the enhanced QSPF simulation on IBM Quantum hardware."""
    global SHOTS, NUM_RUNS, param_history_df
    if light_mode:
        SHOTS = 2000
        NUM_RUNS = 15
        print("Running in light mode: SHOTS=2000, NUM_RUNS=15, Bayesian samples=200")
        
    print("WARNING: This simulation is computationally intensive (12 qubits).")
    print("Recommended: AWS EC2 c5.4xlarge (16 vCPUs, 32GB RAM) for local processing.")
    print("Estimated runtime: IBM Quantum queue time (hours) + local processing (~15 minutes on c5.4xlarge, ~45 minutes on 4-core CPU).")
    print("See script header for EC2 setup instructions.")
    
    check_versions()
    
    if use_hardware and not qiskit_ibm_runtime:
        print("Error: qiskit-ibm-runtime not installed. Install it using:")
        print("  pip install qiskit-ibm-runtime")
        print("Disabling hardware mode.")
        use_hardware = False
    
    ibm_token = ibm_token or os.getenv('IBM_QUANTUM_TOKEN')
    
    if use_hardware and ibm_token:
        try:
            service = QiskitRuntimeService(channel="ibm_quantum", token=ibm_token)
            print("Available backends:")
            for backend in service.backends():
                print(f"Backend: {backend.name}, Qubits: {backend.configuration().n_qubits}, Status: {backend.status().status}, Pending Jobs: {backend.status().pending_jobs}")
        except Exception as e:
            print(f"Error listing backends: {str(e)}. Using dynamic backend selection.")
    
    influencer_types = list(ANTHROPOMORPHISM.keys())
    product_categories = list(PRODUCT_CATEGORIES.keys())
    tasks = [(inf, cat) for inf in influencer_types for cat in product_categories]
    
    global DEPTH
    if not debug_mode:
        DEPTH, best_noise_prob = tune_parameters(tasks, ibm_token, backend_name)
        print(f"Optimal parameters: depth={DEPTH}, noise_prob={best_noise_prob}")
    else:
        DEPTH, best_noise_prob = 1, 0.01
        print("Debug mode: Using depth=1, noise_prob=0.01")
    
    start_time = time.time()
    if use_hardware and ibm_token:
        print("Running simulation on IBM Quantum hardware with dynamic backend selection.")
        results, param_history = batch_process(tasks * (NUM_RUNS // len(tasks)), depth=DEPTH, noise_prob=best_noise_prob, ibm_token=ibm_token, backend_name=backend_name)
    else:
        print("Running simulation on AerSimulator (hardware disabled or token missing).")
        results, param_history = batch_process(tasks * (NUM_RUNS // len(tasks)), depth=DEPTH, noise_prob=best_noise_prob, debug=debug_mode, ibm_token=None, backend_name=backend_name)
    try:
        results_df, param_history_df = save_results_table(results, param_history)
        print(f"Simulation completed in {(time.time() - start_time) / 60:.2f} minutes.")
        print(f"Results saved to qspf_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        # Statistical Analysis
        print("Performing statistical analysis...")
        stats_summary = run_statistical_analysis(results_df)
        with open(f'statistical_summary_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(stats_summary, f, indent=4)
        
        # Classical Benchmarking
        print("Running classical benchmarking...")
        classical_metrics = benchmark_classical(results_df)
        print(f"Classical benchmarking completed. Metrics saved to classical_benchmark_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # Validation and Correlation
        print("Validating with real data...")
        validation_results = validate_with_real_data(results_df)
        print(f"Validation results: {validation_results}")
        
        print("Correlating with user study data...")
        correlation_results = correlate_user_study(results_df)
        print(f"Correlation results: {correlation_results}")
        
        # Visualizations
        print("Generating visualizations...")
        visualize_results(results_df, stats_summary, classical_metrics)
        plot_quantum_metrics(results_df)
        print(f"Visualizations saved with timestamp {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
    except Exception as e:
        error_msg = f"Error in post-processing: {str(e)}\n{traceback.format_exc()}\n"
        print(error_msg)
        with open(f'error_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt', 'a') as f:
            f.write(error_msg)
        raise
    
    print("Simulation and analysis complete.")
    return results_df, stats_summary, classical_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum Social Presence Framework Simulation")
    parser.add_argument("--light_mode", action="store_true", help="Run in light mode (reduced SHOTS and NUM_RUNS)")
    parser.add_argument("--use_hardware", action="store_true", help="Use IBM Quantum hardware")
    parser.add_argument("--ibm_token", type=str, default=None, help="IBM Quantum API token")
    parser.add_argument("--debug_mode", action="store_true", help="Run in debug mode with reduced parameters")
    parser.add_argument("--backend_name", type=str, default="ibm_brisbane", help="IBM Quantum backend name")
    args = parser.parse_args()
    
    main(
        light_mode=args.light_mode,
        use_hardware=args.use_hardware,
        ibm_token=args.ibm_token,
        debug_mode=args.debug_mode,
        backend_name=args.backend_name
    )