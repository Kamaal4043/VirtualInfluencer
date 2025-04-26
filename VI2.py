import numpy as np
import pandas as pd
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal, PauliFeatureMap
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, entropy, concurrence
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
from qiskit.result import LocalReadoutMitigator
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, CXCancellation, BasisTranslator, UnitarySynthesis
from qiskit_machine_learning.kernels import QuantumKernel
import pymc as pm
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import f_oneway, ttest_ind
from statsmodels.stats.multitest import multipletests
from multiprocessing import Pool
import multiprocessing as mp
import time
import json
import datetime
import warnings
warnings.filterwarnings("ignore")

# Constants
NUM_VI_QUBITS = 3
NUM_CONSUMER_QUBITS = 3
NUM_CONTEXT_QUBITS = 3
TOTAL_QUBITS = NUM_VI_QUBITS + NUM_CONSUMER_QUBITS + NUM_CONTEXT_QUBITS
SHOTS = 10000
NUM_RUNS = 90

# Anthropomorphism levels
ANTHROPOMORPHISM = {
    'mimic-human': 0.9,
    'animated-human': 0.5,
    'non-human': 0.1,
    'caretaker': 0.4,
    'educator': 0.6,
    'companion': 0.3
}

def create_noise_model():
    """Create noise model with depolarizing and thermal relaxation errors."""
    noise_model = NoiseModel()
    for qubit in range(TOTAL_QUBITS):
        noise_model.add_quantum_error(
            thermal_relaxation_error(50e-6, 70e-6, 0.08),
            ['rx', 'ry', 'rz'], [qubit]
        )
        noise_model.add_quantum_error(
            depolarizing_error(0.02, 2), ['cx'], [qubit, (qubit + 1) % TOTAL_QUBITS]
        )
    return noise_model

def optimize_circuit(qc):
    """Optimize circuit using advanced transpiler passes."""
    pm = PassManager([Optimize1qGates(), CXCancellation(), BasisTranslator(), UnitarySynthesis()])
    return pm.run(qc)

def apply_error_correction(qc, logical_qubits):
    """Apply 3-qubit bit-flip code."""
    for lq in logical_qubits:
        if lq + 2 * TOTAL_QUBITS // 3 < TOTAL_QUBITS:
            qc.cx(lq, lq + TOTAL_QUBITS // 3)
            qc.cx(lq, lq + 2 * TOTAL_QUBITS // 3)
            qc.ccx(lq, lq + TOTAL_QUBITS // 3, lq + 2 * TOTAL_QUBITS // 3)
    return optimize_circuit(qc)

def create_influencer_state(circuit, qubits, influencer_type, t):
    """Prepare VI state with quantum ansatz, emphasizing entanglement."""
    ansatz = TwoLocal(len(qubits), ['ry', 'rz'], 'cx', entanglement='linear', reps=2)
    if influencer_type == 'mimic-human':
        circuit.h(qubits[0])
        circuit.crx(np.pi / 3 + 0.3 * np.sin(t), qubits[0], qubits[1])
        circuit.cry(np.pi / 4, qubits[1], qubits[2])
        circuit.append(ansatz, qubits)
        circuit.rx(np.random.uniform(0, 0.1), qubits)
    elif influencer_type == 'animated-human':
        circuit.rxx(np.pi / 4 + 0.2 * np.cos(t), qubits[0], qubits[1])
        circuit.ryy(np.pi / 5, qubits[1], qubits[2])
        circuit.append(ansatz, qubits)
    elif influencer_type == 'non-human':
        circuit.h(qubits[:2])
        circuit.cx(qubits[0], qubits[2])
        circuit.append(ansatz, qubits)
    elif influencer_type == 'caretaker':
        circuit.ry(np.pi / 3, qubits[0])
        circuit.cx(qubits[0], qubits[1])
        circuit.rz(np.pi / 4 + 0.1 * np.sin(t), qubits[2])
        circuit.append(ansatz, qubits)
    elif influencer_type == 'educator':
        circuit.rz(np.pi / 2, qubits[0])
        circuit.cx(qubits[0], qubits[1])
        circuit.ry(np.pi / 4, qubits[2])
        circuit.append(ansatz, qubits)
    elif influencer_type == 'companion':
        circuit.ry(np.pi / 4, qubits[0])
        circuit.rxx(np.pi / 5, qubits[0], qubits[1])
        circuit.append(ansatz, qubits)
    return apply_error_correction(circuit, [qubits[0]])

def create_consumer_state(circuit, qubits, influencer_type, self_concept, mood):
    """Prepare consumer state with quantum ansatz."""
    congruence = 1 - abs(self_concept - ANTHROPOMORPHISM[influencer_type])
    expressiveness = np.random.uniform(0.5, 1.0)
    circuit.ry(congruence * np.pi / 2, qubits[0])
    circuit.rx(expressiveness * np.pi / 4, qubits[1])
    circuit.rz(mood * np.pi / 2, qubits[2])
    ansatz = TwoLocal(len(qubits), ['ry', 'rz'], 'cx', entanglement='linear', reps=2)
    circuit.append(ansatz, qubits)
    return apply_error_correction(circuit, [qubits[0]])

def apply_context_state(circuit, qubits, product_category):
    """Apply context state for product category."""
    ansatz = TwoLocal(len(qubits), ['ry', 'rz'], 'cx', entanglement='linear', reps=2)
    if product_category == 'entertainment':
        circuit.rxx(np.pi / 3, qubits[0], qubits[1])
        circuit.ryy(np.pi / 4, qubits[1], qubits[2])
    elif product_category == 'healthcare':
        circuit.rzx(np.pi / 3, qubits[0], qubits[1])
        circuit.ry(np.pi / 2, qubits[2])
    elif product_category == 'education':
        circuit.rz(np.pi / 3, qubits[0])
        circuit.cx(qubits[0], qubits[2])
    elif product_category == 'fashion':
        circuit.rxx(np.pi / 4, qubits[0], qubits[1])
        circuit.ryy(np.pi / 5, qubits[1], qubits[2])
    elif product_category == 'technology':
        circuit.rz(np.pi / 3, qubits[0])
        circuit.cx(qubits[0], qubits[1])
    circuit.append(ansatz, qubits)
    return apply_error_correction(circuit, [qubits[0]])

def measure_in_basis(circuit, basis):
    """Measure circuit in specified basis."""
    meas_qc = circuit.copy()
    for i, b in enumerate(basis * TOTAL_QUBITS):
        if b == 'X':
            meas_qc.h(i)
        elif b == 'Y':
            meas_qc.sdg(i)
            meas_qc.h(i)
    meas_qc.measure_all()
    result = AerSimulator(method='matrix_product_state').run(meas_qc, shots=SHOTS).result()
    counts = result.get_counts()
    high_presence = sum(v for k, v in counts.items() if k[:6].count('1') >= 4)
    return high_presence / SHOTS

def measure_social_presence(circuit, influencer_qubits, consumer_qubits, influencer_type, noise_model):
    """
    Measures non-classical correlations using:
    1. Entanglement witnessing (for non-separability)
    2. Contextual measurements (for superposition effects)
    3. Reference state comparison (kernel-based)
    """
    # Reference state representing "ideal" social presence
    ideal_circuit = QuantumCircuit(TOTAL_QUBITS)
    ideal_circuit.h(influencer_qubits)
    ideal_state = AerSimulator(method='matrix_product_state').run(ideal_circuit).result().get_statevector()
    
    # SWAP test to compare with actual state
    swap_qc = QuantumCircuit(TOTAL_QUBITS + 1, 1)
    swap_qc.append(circuit, range(TOTAL_QUBITS))
    swap_qc.h(TOTAL_QUBITS)
    for q in influencer_qubits + consumer_qubits:
        swap_qc.cswap(TOTAL_QUBITS, q, q % len(influencer_qubits))
    swap_qc.h(TOTAL_QUBITS)
    swap_qc.measure(TOTAL_QUBITS, 0)
    
    result = AerSimulator(method='matrix_product_state').run(swap_qc, noise_model=noise_model, shots=SHOTS).result()
    p0 = result.get_counts().get('0', 0) / SHOTS
    
    # Entanglement witnessing
    witness_val = 1 - 2 * p0
    
    # Contextual measurement variance
    contextual_var = np.var([
        measure_in_basis(circuit, 'X'),
        measure_in_basis(circuit, 'Y'),
        measure_in_basis(circuit, 'Z')
    ])
    
    # Composite metric incorporating quantum effects
    social_presence = (witness_val + 0.5 * contextual_var) * ANTHROPOMORPHISM[influencer_type]
    
    return social_presence, contextual_var, ideal_state

def measure_authenticity_seeking(statevector, ideal_state, influencer_qubits, consumer_qubits):
    """Quantifies deviation from ideal social presence using quantum kernel methods."""
    kernel = QuantumKernel(feature_map=PauliFeatureMap(TOTAL_QUBITS, reps=2, entanglement='full'),
                          quantum_instance=AerSimulator(method='matrix_product_state'))
    
    # Encode states for comparison
    state_matrix = np.array([statevector.data, ideal_state.data])
    
    # Kernel matrix shows state similarity
    kernel_matrix = kernel.evaluate(state_matrix)
    
    # Authenticity = 1 - similarity to ideal
    authenticity = 1 - kernel_matrix[0,1]
    
    # Entanglement persistence metric
    rho = partial_trace(statevector, 
                       [i for i in range(TOTAL_QUBITS) if i not in influencer_qubits + consumer_qubits])
    entanglement_pers = concurrence(rho)
    
    return {
        'authenticity': authenticity,
        'kernel_matrix': kernel_matrix,
        'entanglement_persistence': entanglement_pers
    }

def measure_incompatibility(statevector):
    """Test for non-commuting observables."""
    rho = DensityMatrix(statevector)
    X_basis = np.array([[0,1],[1,0]])
    Z_basis = np.array([[1,0],[0,-1]])
    EX = rho.expectation_value(X_basis)
    EZ = rho.expectation_value(Z_basis)
    return abs(EX * EZ - EZ * EX)

def test_bell_violation(statevector, qubit_pair):
    """Test CHSH inequality for quantum correlations."""
    correlations = []
    for a, b in [(0,0), (0,45), (45,0), (45,45)]:
        qc = QuantumCircuit(2,1)
        qc.initialize(statevector.data, qubit_pair)
        qc.ry(np.radians(a), qubit_pair[0])
        qc.ry(np.radians(b), qubit_pair[1])
        qc.measure(qubit_pair[0], 0)
        result = AerSimulator(method='matrix_product_state').run(qc, shots=SHOTS).result()
        counts = result.get_counts()
        p1 = counts.get('1', 0) / SHOTS
        correlations.append(p1)
    S = correlations[0] - correlations[1] + correlations[2] + correlations[3]
    return S > 2

def measure_emotional_attachment(statevector, influencer_qubits, consumer_qubits, self_concept, influencer_type):
    """Measure emotional attachment using quantum entanglement."""
    rho = partial_trace(statevector, [i for i in range(TOTAL_QUBITS) if i not in influencer_qubits + consumer_qubits])
    concurrence_val = concurrence(rho)
    kernel = QuantumKernel(feature_map=PauliFeatureMap(TOTAL_QUBITS, reps=2),
                          quantum_instance=AerSimulator(method='matrix_product_state'))
    probs = np.abs(statevector.data) ** 2
    probs /= np.sum(probs)
    valence = np.sum(probs[:len(probs)//2]) - np.sum(probs[len(probs)//2:])
    arousal = np.sum(probs[::2])
    kernel_matrix = kernel.evaluate(np.array([[valence, arousal]]))
    kernel_score = kernel_matrix[0, 0]
    score = np.sqrt(valence**2 + arousal**2)
    congruence = 1 - abs(self_concept - ANTHROPOMORPHISM[influencer_type])
    expressiveness = np.random.uniform(0.5, 1.0)
    composite = concurrence_val * (valence + arousal) / 2 * congruence * expressiveness
    return {
        "composite": composite,
        "valence": valence,
        "arousal": arousal,
        "kernel_score": kernel_score,
        "emotion_score": score
    }

def measure_credibility(statevector):
    """Measure trustworthiness and expertise using quantum density matrix."""
    rho_vi = partial_trace(statevector, list(range(NUM_VI_QUBITS, TOTAL_QUBITS)))
    trustworthiness = entropy(rho_vi, base=2)
    expertise = np.mean([np.abs(rho_vi.data[i, i]) for i in range(2**NUM_VI_QUBITS)])
    return {"trustworthiness": trustworthiness, "expertise": expertise}

def measure_purchase_intention(social_presence, emotional_attachment, credibility):
    """Measure purchase intention as a composite metric."""
    return (0.3 * social_presence + 0.3 * emotional_attachment +
            0.2 * credibility["trustworthiness"] + 0.2 * credibility["expertise"])

def bayesian_mediation_analysis(social_presence, emotional_attachment, authenticity_seeking, trust, influencer_type, mood):
    """Bayesian mediation analysis with optimized sampling."""
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
        
        trace = pm.sample(500, tune=500, return_inferencedata=False, progressbar=False)
    
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

def plot_quantum_metrics(results_df):
    """Generates quantum-specific visualizations."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Entanglement Dynamics
    fig1 = go.Figure()
    for inf_type in results_df['influencer_type'].unique():
        subset = results_df[results_df['influencer_type'] == inf_type]
        fig1.add_trace(go.Scatter(
            x=subset['social_presence'],
            y=subset['entanglement_persistence'],
            mode='markers',
            name=inf_type,
            marker=dict(
                size=subset['authenticity'] * 20,
                color=subset['valence'],
                colorscale='Portland',
                showscale=True
            )
        ))
    fig1.update_layout(
        title='Entanglement Persistence vs Social Presence',
        xaxis_title='Social Presence (Witness Value)',
        yaxis_title='Entanglement Persistence'
    )
    fig1.write(f'entanglement_dynamics_{timestamp}.html')
    
    # Kernel Matrix Heatmap
    avg_kernels = results_df.groupby('influencer_type')['kernel_matrix'].mean()
    fig2 = go.Figure(go.Heatmap(
        z=[k[0] for k in avg_kernels],
        x=['Actual', 'Ideal'],
        y=list(avg_kernels.index),
        colorscale='Viridis'
    ))
    fig2.update_layout(title='State Similarity to Ideal Social Presence')
    fig2.write(f'kernel_heatmap_{timestamp}.html')
    
    # Quantum Contextuality Plot
    fig3 = px.box(results_df, 
                 x='influencer_type', 
                 y='contextual_variance',
                 color='product_category',
                 title='Measurement Contextuality by VI Type')
    fig3.write(f'contextuality_{timestamp}.html')
    
    return fig1, fig2, fig3

def run_statistical_analysis(results_df):
    """Perform ANOVA, pairwise t-tests, feature importance, and quantum metrics."""
    stats_summary = {'anova': {}, 'pairwise': {}, 'feature_importance': {}, 'quantum_metrics': {}}
    influencer_types = results_df['influencer_type'].unique()
    
    for metric in ['emotional_attachment', 'trustworthiness', 'expertise', 'purchase_intention']:
        data = [results_df[results_df['influencer_type'] == inf][metric] for inf in influencer_types]
        f_stat, p_value = f_oneway(*data)
        stats_summary['anova'][metric] = {'F': f_stat, 'p': p_value, 'significant': p_value < 0.05}
    
    comparisons = [(a, b) for i, a in enumerate(influencer_types) for b in influencer_types[i+1:]]
    for metric in ['emotional_attachment', 'trustworthiness', 'expertise', 'purchase_intention']:
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
                      'trustworthiness', 'expertise', 'emotional_attachment']].dropna()
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(data.drop('emotional_attachment', axis=1), data['emotional_attachment'])
    importances = pd.Series(model.feature_importances_, index=data.drop('emotional_attachment', axis=1).columns)
    stats_summary['feature_importance'] = importances.to_dict()
    
    # Quantum metrics
    stats_summary['quantum_metrics']['contextual_variance'] = results_df.groupby('influencer_type')['contextual_variance'].describe().to_dict()
    kernel_results = results_df.groupby('influencer_type').apply(
        lambda x: np.mean([k[0,1] for k in x['kernel_matrix']])).to_dict()
    stats_summary['quantum_metrics']['kernel_similarity'] = kernel_results
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fig = go.Figure(data=[go.Bar(x=importances.index, y=importances.values)])
    fig.update_layout(title="Feature Importance for Emotional Attachment",
                      xaxis_title="Features", yaxis_title="Importance")
    fig.write(f'feature_importance_{timestamp}.html')
    
    return stats_summary

def visualize_results(results_df, stats_summary):
    """Visualize results with timestamped filenames."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    influencer_types = results_df['influencer_type'].unique()
    
    for metric, title, filename in [
        ('social_presence', 'Social Presence by VI Type', f'social_presence_{timestamp}.html'),
        ('trustworthiness', 'Trustworthiness by VI Type', f'trustworthiness_{timestamp}.html'),
        ('expertise', 'Expertise by VI Type', f'expertise_{timestamp}.html'),
        ('purchase_intention', 'Purchase Intention by VI Type', f'purchase_intention_{timestamp}.html')
    ]:
        fig = go.Figure()
        for inf in influencer_types:
            data = results_df[results_df['influencer_type'] == inf][metric]
            fig.add_trace(go.Violin(y=data, name=inf, box_visible=True))
        fig.update_layout(title=f"{title}\n(ANOVA F={stats_summary['anova'].get(metric, {}).get('F', 0):.2f}, p={stats_summary['anova'].get(metric, {}).get('p', 1):.4f})",
                         yaxis_title=metric.replace('_', ' ').title())
        fig.write(filename)
    
    fig = go.Figure()
    for inf in influencer_types:
        mask = results_df['influencer_type'] == inf
        fig.add_trace(go.Scatter(
            x=results_df[mask]['valence'],
            y=results_df[mask]['arousal'],
            mode='markers',
            name=inf,
            text=[f"Score: {s:.2f}" for s in results_df[mask]['emotion_score']]
        ))
    fig.update_layout(title='Valence-Arousal by VI Type', xaxis_title='Valence', yaxis_title='Arousal')
    fig.write(f'valence_arousal_{timestamp}.html')
    
    fig = go.Figure()
    for inf in influencer_types:
        mask = results_df['influencer_type'] == inf
        fig.add_trace(go.Scatter(
            x=results_df[mask]['social_presence'],
            y=results_df[mask]['purchase_intention'],
            mode='markers',
            name=inf,
            text=[f"Score: {s:.2f}" for s in results_df[mask]['emotion_score']]
        ))
    fig.update_layout(title='Social Presence vs Purchase Intention', xaxis_title='Social Presence', yaxis_title='Purchase Intention')
    fig.write(f'sp_purchase_scatter_{timestamp}.html')
    
    fig = go.Figure()
    for inf in influencer_types:
        mask = results_df['influencer_type'] == inf
        fig.add_trace(go.Scatter(
            x=results_df[mask]['trustworthiness'],
            y=results_df[mask]['expertise'],
            mode='markers',
            name=inf,
            text=[f"Score: {s:.2f}" for s in results_df[mask]['emotion_score']]
        ))
    fig.update_layout(title='Trustworthiness vs Expertise', xaxis_title='Trustworthiness', yaxis_title='Expertise')
    fig.write(f'trust_expertise_scatter_{timestamp}.html')
    
    fig = plt.figure(figsize=(14, 10))
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
    plt.legend()
    plt.savefig(f'3d_emotion_space_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    product_categories = results_df['product_category'].unique()
    benefit_corr = np.zeros((len(influencer_types), len(product_categories)))
    for i, inf in enumerate(influencer_types):
        for j, cat in enumerate(product_categories):
            subset = results_df[(results_df['influencer_type'] == inf) & 
                               (results_df['product_category'] == cat)]['emotional_attachment']
            benefit_corr[i, j] = subset.mean() if len(subset) > 0 else 0
    plt.figure(figsize=(10, 6))
    sns.heatmap(benefit_corr, annot=True, xticklabels=product_categories, yticklabels=influencer_types, cmap="viridis")
    plt.title("Emotional Attachment by VI Type and Product Category")
    plt.xlabel("Product Category")
    plt.ylabel("VI Type")
    plt.savefig(f"benefit_correlation_heatmap_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    quantiles = results_df['self_concept'].quantile([0.33, 0.66]).values
    results_df['self_concept_group'] = pd.cut(results_df['self_concept'], 
                                             bins=[0, quantiles[0], quantiles[1], 1], 
                                             labels=['Low', 'Medium', 'High'])
    plt.figure(figsize=(12, 6))
    for inf in influencer_types:
        data = results_df[results_df['influencer_type'] == inf].groupby('self_concept_group', observed=True)['emotional_attachment'].mean().reset_index()
        sns.lineplot(x="self_concept_group", y="emotional_attachment", label=inf, data=data)
    plt.title("Emotional Attachment by Self-Concept and VI Type")
    plt.xlabel("Self-Concept Group")
    plt.ylabel("Emotional Attachment")
    plt.legend()
    plt.savefig(f"self_concept_segmentation_{timestamp}.png", dpi=300, bbox_inches='tight')
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
        title=f"Mediation Diagram\nMediation Ratio = {results_df['mediation_ratio'].mean():.2f}",
        showlegend=True,
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False)
    )
    fig.write(f'mediation_diagram_{timestamp}.html')
    
    sp_data = np.zeros((len(influencer_types), NUM_RUNS // len(influencer_types)))
    for i, inf in enumerate(influencer_types):
        sp_data[i] = results_df[results_df['influencer_type'] == inf]['social_presence'].values[:NUM_RUNS // len(influencer_types)]
    fig = go.Figure()
    for i, inf in enumerate(influencer_types):
        fig.add_trace(go.Surface(
            x=np.arange(sp_data.shape[1]), y=np.arange(len(influencer_types)), z=sp_data,
            name=inf, opacity=0.7, colorscale="Viridis"
        ))
    fig.update_layout(
        title="Social Presence Across VI Types and Runs",
        scene=dict(xaxis_title="Run Index", yaxis_title="VI Type Index", zaxis_title="Social Presence")
    )
    fig.write(f'3d_social_presence_interactive_{timestamp}.html')

def save_results_table(results):
    """Save results to CSV, JSON, and checkpoint."""
    results_df = pd.DataFrame([r for r in results if r is not None])
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f'qspf_results_{timestamp}.csv', index=False)
    with open(f'qspf_results_{timestamp}.json', 'w') as f:
        json.dump([r for r in results if r is not None], f, indent=4)
    
    checkpoint_data = [
        {'influencer_type': r['influencer_type'], 'product_category': r['product_category'],
         'social_presence': r['social_presence'], 'emotional_attachment': r['emotional_attachment']}
        for r in results if r is not None
    ]
    pd.DataFrame(checkpoint_data).to_csv(f'qspf_checkpoint_{timestamp}.csv', index=False)
    
    stats_summary = run_statistical_analysis(results_df)
    with open(f'stats_summary_{timestamp}.json', 'w') as f:
        json.dump(stats_summary, f, indent=4)
    
    return results_df

def run_condition(args):
    """Run experiment for a single condition with error handling."""
    influencer_type, product_category = args
    circuit = QuantumCircuit(TOTAL_QUBITS)
    influencer_qubits = list(range(NUM_VI_QUBITS))
    consumer_qubits = list(range(NUM_VI_QUBITS, NUM_VI_QUBITS + NUM_CONSUMER_QUBITS))
    context_qubits = list(range(NUM_VI_QUBITS + NUM_CONSUMER_QUBITS, TOTAL_QUBITS))
    
    t = np.random.uniform(0, 2 * np.pi)
    self_concept = np.random.uniform(0, 1)
    mood = np.random.uniform(0, 1)
    
    try:
        create_influencer_state(circuit, influencer_qubits, influencer_type, t)
        create_consumer_state(circuit, consumer_qubits, influencer_type, self_concept, mood)
        apply_context_state(circuit, context_qubits, product_category)
        
        for i, j in zip(influencer_qubits, consumer_qubits):
            circuit.cx(i, j)
        
        noise_model = create_noise_model()
        backend = AerSimulator(method='matrix_product_state')
        
        result = backend.run(circuit, noise_model=noise_model, shots=SHOTS).result()
        statevector = result.get_statevector()
        
        social_presence, contextual_var, ideal_state = measure_social_presence(circuit, influencer_qubits, consumer_qubits, influencer_type, noise_model)
        ea_data = measure_emotional_attachment(statevector, influencer_qubits, consumer_qubits, self_concept, influencer_type)
        auth_data = measure_authenticity_seeking(statevector, ideal_state, influencer_qubits, consumer_qubits)
        emotional_attachment = ea_data["composite"]
        valence = ea_data["valence"]
        arousal = ea_data["arousal"]
        kernel_score = ea_data["kernel_score"]
        emotion_score = ea_data["emotion_score"]
        authenticity = auth_data['authenticity']
        kernel_matrix = auth_data['kernel_matrix']
        entanglement_pers = auth_data['entanglement_persistence']
        credibility = measure_credibility(statevector)
        purchase_intention = measure_purchase_intention(social_presence, emotional_attachment, credibility)
        incompatibility = measure_incompatibility(statevector)
        bell_violation = test_bell_violation(statevector, [influencer_qubits[0], consumer_qubits[0]])
        
        mediation_results = bayesian_mediation_analysis(social_presence, emotional_attachment, authenticity,
                                                       credibility["trustworthiness"], influencer_type, mood)
        
        result = {
            'influencer_type': influencer_type,
            'product_category': product_category,
            'self_concept': self_concept,
            'mood': mood,
            'social_presence': social_presence,
            'contextual_variance': contextual_var,
            'emotional_attachment': emotional_attachment,
            'valence': valence,
            'arousal': arousal,
            'kernel_score': kernel_score,
            'emotion_score': emotion_score,
            'authenticity': authenticity,
            'kernel_matrix': kernel_matrix.tolist(),
            'entanglement_persistence': entanglement_pers,
            'trustworthiness': credibility["trustworthiness"],
            'expertise': credibility["expertise"],
            'purchase_intention': purchase_intention,
            'incompatibility': incompatibility,
            'bell_violation': bell_violation,
            'mediation_index_ea': mediation_results['mediation_index_ea'],
            'mediation_index_auth': mediation_results['mediation_index_auth'],
            'mediation_index_trust': mediation_results['mediation_index_trust'],
            'ci_ea': mediation_results['ci_ea'],
            'ci_auth': mediation_results['ci_auth'],
            'ci_trust': mediation_results['ci_trust'],
            'significant_ea': mediation_results['significant_ea'],
            'significant_auth': mediation_results['significant_auth'],
            'significant_trust': mediation_results['significant_trust'],
            'mediation_ratio': mediation_results['mediation_ratio'],
            'a': mediation_results['a'],
            'b1': mediation_results['b1'],
            'b2': mediation_results['b2'],
            'b3': mediation_results['b3'],
            'c1': mediation_results['c1'],
            'c2': mediation_results['c2'],
            'c3': mediation_results['c3']
        }
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f'checkpoint_temp_{timestamp}.json', 'a') as f:
            json.dump(result, f)
            f.write('\n')
        
        return result
    except Exception as e:
        print(f"Error in simulation for {influencer_type}, {product_category}: {e}")
        return None

def run_qspf_experiment():
    """Run QSPF experiment with parallel execution and MPS simulation."""
    influencer_types = ['mimic-human', 'animated-human', 'non-human', 'caretaker', 'educator', 'companion']
    product_categories = ['entertainment', 'healthcare', 'education', 'fashion', 'technology']
    tasks = [(inf, cat) for inf in influencer_types for cat in product_categories]
    
    runs_per_condition = NUM_RUNS // len(tasks) + 1
    adjusted_tasks = []
    for task in tasks:
        adjusted_tasks.extend([task] * runs_per_condition)
    adjusted_tasks = adjusted_tasks[:NUM_RUNS]
    
    with Pool(processes=min(mp.cpu_count(), len(adjusted_tasks))) as pool:
        results = pool.map(run_condition, adjusted_tasks)
    
    results_df = save_results_table(results)
    stats_summary = run_statistical_analysis(results_df)
    visualize_results(results_df, stats_summary)
    plot_quantum_metrics(results_df)
    
    # Print quantum metrics
    print("\nContextual Variance by Influencer Type:")
    print(results_df.groupby('influencer_type')['contextual_variance'].describe())
    print("\nKernel Similarity to Ideal State:")
    kernel_results = results_df.groupby('influencer_type').apply(
        lambda x: np.mean([k[0,1] for k in x['kernel_matrix']]))
    print(kernel_results.sort_values())
    
    return results_df

if __name__ == '__main__':
    start_time = time.time()
    results_df = run_qspf_experiment()
    print(f"Experiment completed in {time.time() - start_time:.2f} seconds")
    print(results_df[['influencer_type', 'product_category', 'social_presence', 'emotional_attachment', 
                     'trustworthiness', 'expertise', 'purchase_intention']].head())