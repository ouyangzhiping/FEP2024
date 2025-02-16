import numpy as np
import scipy.stats as stats
from scipy.special import logsumexp, gammaln, psi
from utility.math_utils import nat_log
from spm.spm_auxillary import spm_wnorm


def explore_exploit_model(Gen_model):
    # Number of time points or 'epochs' within a trial: T
    T = 3

    # Priors about initial states: D and d
    D = {}
    D[1] = np.array([[1], [0]])  # {'left better','right better'}
    D[2] = np.array([[1], [0], [0], [0]])  # {'start','hint','choose-left','choose-right'}

    d = {}
    d[1] = np.array([[0.25], [0.25]])  # {'left better','right better'}
    d[2] = np.array([[1], [0], [0], [0]])  # {'start','hint','choose-left','choose-right'}

    # State-outcome mappings and beliefs: A and a
    Ns = [len(D[1]), len(D[2])]  # number of states in each state factor (2 and 4)

    A = {}
    A[1] = np.zeros((3, 2, 4))
    for i in range(Ns[1]):
        A[1][:, :, i] = np.array([[1, 1], [0, 0], [0, 0]])

    pHA = 1
    A[1][:, :, 1] = np.array([[0, 0], [pHA, 1 - pHA], [1 - pHA, pHA]])

    A[2] = np.zeros((3, 2, 4))
    for i in range(2):
        A[2][:, :, i] = np.array([[1, 1], [0, 0], [0, 0]])

    pWin = 0.8
    A[2][:, :, 2] = np.array([[0, 0], [1 - pWin, pWin], [pWin, 1 - pWin]])
    A[2][:, :, 3] = np.array([[0, 0], [pWin, 1 - pWin], [1 - pWin, pWin]])

    A[3] = np.zeros((4, 2, 4))
    for i in range(Ns[1]):
        A[3][i, :, i] = np.array([1, 1])

    a = {}
    a[1] = A[1] * 200
    a[2] = A[2] * 200
    a[3] = A[3] * 200
    a[1][:, :, 1] = np.array([[0, 0], [0.25, 0.25], [0.25, 0.25]])

    # Controlled transitions and transition beliefs : B{:,:,u} and b(:,:,u)
    B = {}
    B[1] = np.zeros((2, 2, 1))
    B[1][:, :, 0] = np.array([[1, 0], [0, 1]])

    B[2] = np.zeros((4, 4, 4))
    B[2][:, :, 0] = np.array([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    B[2][:, :, 1] = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
    B[2][:, :, 2] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]])
    B[2][:, :, 3] = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]])

    # Preferred outcomes: C and c
    No = [A[1].shape[0], A[2].shape[0], A[3].shape[0]]

    C = {}
    C[1] = np.zeros((No[0], T))
    C[2] = np.zeros((No[1], T))
    C[3] = np.zeros((No[2], T))

    la = 1
    rs = 4
    C[2][:, :] = np.array([[0, 0, 0], [0, -la, -la], [0, rs, rs / 2]])

    # Allowable policies: U or V.
    NumPolicies = 5
    NumFactors = 2

    V = np.ones((T - 1, NumPolicies, NumFactors))
    V[:, :, 0] = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    V[:, :, 1] = np.array([[1, 2, 2, 3, 4], [1, 3, 4, 1, 1]])

    # Habits: E and e.
    E = np.array([[1, 1, 1, 1, 1]]).T
    e = np.array([[1, 1, 1, 1, 1]]).T

    # Additional optional parameters.
    eta = 1
    omega = 1
    beta = 1
    alpha = 32

    # Define POMDP Structure
    mdp = {
        'T': T,
        'V': V,
        'A': A,
        'B': B,
        'C': C,
        'D': D,
        'd': d,
        'eta': eta,
        'omega': omega,
        'alpha': alpha,
        'beta': beta,
        'NumPolicies': NumPolicies,
        'NumFactors': NumFactors
    }

    if Gen_model == 1:
        mdp['E'] = E
    elif Gen_model == 2:
        mdp['a'] = a
        mdp['e'] = e

    # Labels for states, outcomes, and actions
    label = {
        'factor': {1: 'contexts', 2: 'choice states'},
        'name': {1: ['left-better', 'right-better'], 2: ['start', 'hint', 'choose left', 'choose right']},
        'modality': {1: 'hint', 2: 'win/lose', 3: 'observed action'},
        'outcome': {1: ['null', 'left hint', 'right hint'], 2: ['null', 'lose', 'win'], 3: ['start', 'hint', 'choose left', 'choose right']},
        'action': {2: ['start', 'hint', 'left', 'right']}
    }
    mdp['label'] = label

    return mdp

def col_norm(input_dict):
    normalized_dict = {}  # Initialize a dictionary to store the normalized arrays
    
    for key, array in input_dict.items():
        normalized_array = array.copy()  # Make a copy of the original array
        z = np.sum(normalized_array, axis=0)  # Create normalizing constant from the sum of columns
        normalized_array = normalized_array / z  # Divide columns by the constant
        normalized_dict[key] = normalized_array  # Store the normalized array in the dictionary
    
    return normalized_dict

def flatten_3d_to_2d(x):
    if x.ndim != 3:
        raise ValueError("Input array must be 3-dimensional")
    return x.transpose(2, 1, 0).reshape(-1, x.shape[0])

def md_dot(A, s, f):
    if f == 0:
        B = np.dot(A.T, s)
    elif f == 1:
        B = np.dot(A, s)
    else:
        raise ValueError("f must be either 0 or 1.")
    
    return B

def cell_md_dot(X, x):
    # Initialize dimensions
    DIM = np.arange(len(x)) + X.ndim - len(x)
    XNDIM = X.ndim
    # Compute dot product
    for d in range(len(x)):
        s = np.ones(XNDIM, dtype=int)
        s[DIM[d]] = len(x[d])
        X = X * np.reshape(np.array(x[d]), s)
        X = np.sum(X, axis=DIM[d])
    
    X = np.squeeze(X)
    return X

def G_epistemic_value(A, s):
    """
    Auxiliary function for Bayesian surprise or mutual information.

    Parameters:
    A   - likelihood array (probability of outcomes given causes)
    s   - probability density of causes

    Returns:
    G   - epistemic value
    """

    # Probability distribution over the hidden causes: i.e., Q(s)
    qx = spm_cross(s)  # This is the outer product of the posterior over states
                       # calculated with respect to itself

    # Accumulate expectation of entropy: i.e., E[lnP(o|s)]
    G = 0
    qo = np.array([0])  # Initialize qo with zeros

    qx = qx.T.flatten()  # Transpose qx to match the original MATLAB code
    for i in np.where(qx > np.exp(-16))[0]:
    # for i in np.ndindex(np.where(qx > np.exp(-16))):
        # Probability over outcomes for this combination of causes
        po = 1
        for g in range(len(A)):
            po = spm_cross(po, flatten_3d_to_2d(A[g])[i])
        po = po.flatten()
        # qo = qo + qx.flatten()[i] * po
        # G = G + qx.flatten()[i] * np.dot(po, nat_log(po))
        qo = qo + qx[i] * po
        G = G + qx[i] * np.dot(po, nat_log(po))

    # Subtract entropy of expectations: i.e., E[lnQ(o)]
    G = G - np.dot(qo, nat_log(qo))

    return G

def spm_cross(X, x=None, *args):
    # Handle single input
    if x is None:
        if isinstance(X, np.ndarray):
            Y = X
        else:
            Y = spm_cross(*X)
        return Y

    # Handle cell arrays (lists in Python)
    if isinstance(X, list):
        X = spm_cross(*X)
    if isinstance(x, list):
        x = spm_cross(*x)

    # Outer product of first pair of arguments
    if isinstance(X, int):
        A = X
        B = np.reshape(x, (1,) * 1 + x.shape)
    else:
        A = np.reshape(X, X.shape + (1,) * x.ndim)
        B = np.reshape(x, (1,) * X.ndim + x.shape)
    Y = np.squeeze(A * B)

    # Handle remaining arguments
    for arg in args:
        Y = spm_cross(Y, arg)

    return Y

def spm_KL_dir(q, p):
    """
    KL divergence between two Dirichlet distributions
    Calculate KL(Q||P) = <log Q/P> where avg is wrt Q between two Dirichlet distributions Q and P

    Parameters:
    q : array-like
        Concentration parameter matrix of Q
    p : array-like
        Concentration parameter matrix of P

    Returns:
    d : float
        The KL divergence between Q and P
    """
    # KL divergence based on log beta functions
    d = spm_betaln(p) - spm_betaln(q) - np.sum((p - q) * spm_psi(q + 1/32), axis=0)
    d = np.sum(d)

    return d

# def spm_betaln(z):
#     """
#     Returns the log of the multivariate beta function of a vector.
    
#     Parameters:
#     z (array-like): Input vector or array.
    
#     Returns:
#     y (float or ndarray): The natural logarithm of the beta function for corresponding elements of the vector z.
#     """
#     if np.ndim(z) == 1:
#         z = z[np.nonzero(z)]
#         y = np.sum(gammaln(z)) - gammaln(np.sum(z))
#     else:
#         y = np.zeros((1,) + z.shape[1:])
#         for i in range(z.shape[1]):
#             for j in range(z.shape[2]):
#                 for k in range(z.shape[3]):
#                     for l in range(z.shape[4]):
#                         for m in range(z.shape[5]):
#                             y[0, i, j, k, l, m] = spm_betaln(z[:, i, j, k, l, m])
#     return y

def spm_betaln(z):
    """
    Returns the log of the multivariate beta function of a vector.
    
    Parameters:
    z (array-like): Input vector or array.
    
    Returns:
    y (float or ndarray): The natural logarithm of the beta function for corresponding elements of the vector z.
    """
    if np.ndim(z) > 1:
        z = z[np.nonzero(z)]
        y = np.sum(gammaln(z)) - gammaln(np.sum(z))
    else:
        y = np.zeros(z.shape[1:])
        it = np.nditer(y, flags=['multi_index'], op_flags=['writeonly'])
        while not it.finished:
            idx = it.multi_index
            y[idx] = spm_betaln(z[(slice(None),) + idx])
            it.iternext()
    return y

def spm_psi(A):
    """
    Normalization of a probability transition rate matrix (columns)
    :param A: numeric array
    :return: normalized array
    """
    return psi(A) - psi(np.sum(A, axis=0))

def B_norm(B):
    bb = B.copy()  # Create a copy of B to avoid modifying the original
    z = np.sum(bb, axis=0)  # Create normalizing constant from sum of columns
    bb = bb / z  # Divide columns by constant
    bb[np.isnan(bb)] = 0  # Replace NaN with zero
    return bb

# Random seed initialization
np.random.seed()

# Simulation Settings
Gen_model = 1  # As in the main tutorial code

# Specify Generative Model
MDP = explore_exploit_model(Gen_model)  # Placeholder for the model function

# Normalize generative process and generative model
A = MDP['A']  # Likelihood matrices
B = MDP['B']  # Transition matrices
C = MDP['C']  # Preferences over outcomes
D = MDP['D']  # Priors over initial states
T = MDP['T']  # Time points per trial
V = MDP['V']  # Policies
beta = MDP['beta']  # Expected free energy precision
alpha = MDP['alpha']  # Action precision
eta = MDP['eta']  # Learning rate
omega = MDP['omega']  # Forgetting rate

A = col_norm(A)
B = col_norm(B)
D = col_norm(D)

# Generative model (lowercase matrices/vectors are beliefs about capitalized matrices/vectors)
NumPolicies = MDP['NumPolicies']  # Number of policies
NumFactors = MDP['NumFactors']  # Number of state factors

# Store initial parameter values of generative model for free energy calculations after learning
if 'd' in MDP:
    d_prior = {}
    d_complexity = {}
    for factor in range(len(MDP['d'])):
        d_prior[factor + 1] = MDP['d'][factor + 1]
        d_complexity[factor + 1] = spm_wnorm(d_prior[factor+1])

if 'a' in MDP:
    a_prior = {}
    a_complexity = {}
    for modality in range(len(MDP['a'])):
        a_prior[modality] = MDP['a'][modality]
        a_complexity[modality] = spm_wnorm(a_prior[modality]) * (a_prior[modality] > 0)

# Normalize matrices before model inversion/inference
if 'a' in MDP:
    a = col_norm(MDP['a'])
else:
    a = col_norm(MDP['A'])

if 'b' in MDP:
    b = col_norm(MDP['b'])
else:
    b = col_norm(MDP['B'])

for ii in range(len(C)):
    C[ii] = MDP['C'][ii + 1] + 1 / 32
    for t in range(T):
        C[ii][:, t] = nat_log(np.exp(C[ii][:, t]) / np.sum(np.exp(C[ii][:, t])))

if 'd' in MDP:
    d = col_norm(MDP['d'])
else:
    d = col_norm(MDP['D'])

if 'e' in MDP:
    E = MDP['e']
    E = E / np.sum(E)
elif 'E' in MDP:
    E = MDP['E']
    E = E / np.sum(E)
else:
    E = col_norm(np.ones((NumPolicies, 1)))
    E = E / np.sum(E)

# Initialize variables
NumModalities = len(a)  # Number of outcome factors
NumFactors = len(d)  # Number of hidden state factors
NumPolicies = V.shape[1]  # Number of allowable policies
NumStates = np.zeros(NumFactors, dtype=int)
NumControllable_transitions = np.zeros(NumFactors, dtype=int)

for factor in range(NumFactors):
    NumStates[factor] = b[factor + 1].shape[0]
    NumControllable_transitions[factor] = b[factor + 1].shape[2]

# Initialize the approximate posterior over states conditioned on policies
state_posterior = {}
for policy in range(NumPolicies):
    for factor in range(NumFactors):
        NumStates[factor] = len(D[factor + 1])
        state_posterior[factor] = np.ones((NumStates[factor], T, policy + 1)) / NumStates[factor]

# Initialize the approximate posterior over policies
policy_posteriors = np.ones((NumPolicies, T)) / NumPolicies

# Initialize posterior over actions
chosen_action = np.zeros((len(B), T - 1), dtype=int)

# If there is only one policy
for factors in range(NumFactors):
    if NumControllable_transitions[factors] == 1:
        chosen_action[factors, :] = np.ones(T - 1)

MDP['chosen_action'] = chosen_action

# Initialize expected free energy precision (beta)
posterior_beta = 1
gamma = [1 / posterior_beta] * np.ones(T)  # Expected free energy precision

# Message passing variables
TimeConst = 4  # Time constant for gradient descent
NumIterations = 16  # Number of message passing iterations


# Lets go! Message passing and policy selection 
#--------------------------------------------------------------------------
# Initialize necessary variables
true_states = np.zeros((NumFactors, T))
outcomes = np.zeros((NumModalities, T))
O = {}
Ft = np.zeros((T, NumIterations, T, NumFactors))
F = np.zeros((NumPolicies, T))
G = np.zeros((NumPolicies, T))
policy_priors = np.zeros((NumPolicies, T))
policy_posteriors = np.zeros((NumPolicies, T))
gamma_update = np.zeros((NumIterations * T, 1))
policy_posterior_updates = np.zeros((NumPolicies, NumIterations * T))
policy_posterior = np.zeros((NumPolicies, T))
BMA_states = {}
action_posterior = np.zeros((1, NumControllable_transitions[-1], T - 1))

normalized_firing_rates = {}
prediction_error = {}
Expected_states = {}
for factor in range(NumFactors):
    # normalized_firing_rates = np.array([np.zeros((LEN_ITER, 2, 3, 3, 5)), np.zeros((LEN_ITER, 4, 3, 3, 5))], dtype=object)
    normalized_firing_rates[factor] = np.zeros((NumIterations, NumStates[factor], T, T, NumPolicies))
    # prediction_error = np.array([np.zeros((16, 2, 3, 3, 5)), np.zeros((16, 4, 3, 3, 5))], dtype=object)
    prediction_error[factor] = np.zeros((NumIterations, NumStates[factor], T, T, NumPolicies))
    # Expected_states = np.array([np.zeros((2, 1)), np.zeros((4, 1))], dtype=object)
    Expected_states[factor] = np.zeros((NumStates[factor]))

# Main loop
for t in range(T):
    for factor in range(NumFactors):
        if t == 0:
            # Sample initial states
            prob_state = D[factor + 1]
        else:
            prob_state = B[factor + 1][:, true_states[factor, t-1], MDP['chosen_action'][factor, t-1] - 1]
        true_states[factor, t] = np.argmax(np.cumsum(prob_state) >= np.random.rand())
    
    # change the dtype for index calculation
    true_states = np.array(true_states, dtype=int)

    for modality in range(NumModalities):
        outcomes[modality, t] = np.argmax(np.cumsum(a[modality + 1][:, true_states[0, t], true_states[1, t]]) >= np.random.rand())
    for modality in range(NumModalities):
        vec = np.zeros((1, a[modality + 1].shape[0]))
        index = int(outcomes[modality, t])
        vec[0, index] = 1
        O[(modality, t)] = vec

    for policy in range(NumPolicies):
        for Ni in range(NumIterations):
            for factor in range(NumFactors):
                lnAo = np.zeros_like(state_posterior[factor])
                for tau in range(T):
                    v_depolarization = nat_log(state_posterior[factor][:, tau, policy])
                    if tau < t + 1:
                        for modal in range(NumModalities):
                            # TODO: different from original matlab code, because we can have redundant dimensions in Matlab, but not applicable in Python...
                            # ...No impact on the result.
                            lnA = nat_log(a[modal + 1][int(outcomes[modal, tau]), :, :])
                            for fj in range(NumFactors):
                                if fj != factor:
                                    # TODO: there may be an issue in the original m code that the dimension "policy" of lnAs is missing. Use fixed 0 instead.
                                    lnAs = md_dot(lnA, state_posterior[fj][:, tau, 0], fj)
                                    lnA = lnAs
                            # TODO: there may be an issue in the original m code that the dimension "policy" of lnAo is missing. Use fixed 0 instead.
                            lnAo[:, tau, 0] += lnA
                    if tau == 0:
                        lnD = nat_log(d[factor + 1])
                        lnBs = nat_log(B_norm(b[factor + 1][:, :, int(V[tau, policy, factor] - 1)].T) @ state_posterior[factor][:, tau + 1, policy])
                    elif tau == T - 1:
                        lnD = nat_log(b[factor + 1][:, :, int(V[tau - 1, policy, factor] - 1)] @ state_posterior[factor][:, tau - 1, policy])
                        lnBs = np.zeros_like(d[factor + 1])
                    else:
                        lnD = nat_log(b[factor + 1][:, :, int(V[tau - 1, policy, factor] - 1)] @ state_posterior[factor][:, tau - 1, policy])
                        lnBs = nat_log(B_norm(b[factor + 1][:, :, int(V[tau, policy, factor] -1)].T) @ state_posterior[factor][:, tau + 1, policy])
                    # TODO: there may be an issue in the original m code that the dimension "policy" of lnAo is missing. Use fixed 0 instead.
                    # v_depolarization += (0.5 * lnD.reshape(v_depolarization.shape) + 0.5 * lnBs.reshape(v_depolarization.shape) + lnAo[:, tau, 0] - v_depolarization) / TimeConst
                    v_depolarization += (0.5 * lnD.reshape(v_depolarization.shape) + 0.5 * lnBs.reshape(v_depolarization.shape) + flatten_3d_to_2d(lnAo)[tau] - v_depolarization) / TimeConst
                    # TODO: there may be an issue in the original m code that the dimension "policy" of lnAo is missing. Use fixed 0 instead.
                    # Ft[tau, Ni, t, factor] = state_posterior[factor][:, tau, policy].T @ (0.5 * lnD.reshape(v_depolarization.shape) + 0.5 * lnBs.reshape(v_depolarization.shape) + lnAo[:, tau, 0] - nat_log(state_posterior[factor][:, tau, policy]))
                    Ft[tau, Ni, t, factor] = state_posterior[factor][:, tau, policy].T @ (0.5 * lnD.reshape(v_depolarization.shape) + 0.5 * lnBs.reshape(v_depolarization.shape) + flatten_3d_to_2d(lnAo)[tau] - nat_log(state_posterior[factor][:, tau, policy]))
                    state_posterior[factor][:, tau, policy] = np.exp(v_depolarization) / np.sum(np.exp(v_depolarization))
                    normalized_firing_rates[factor][Ni, :, tau, t, policy] = state_posterior[factor][:, tau, policy]
                    prediction_error[factor][Ni, :, tau, t, policy] = v_depolarization
        Fintermediate = np.sum(Ft, axis=3)
        # TODO: this is a patch to adjust the size of Fintermediate. Could be optimized.
        Fintermediate = Fintermediate[:,:,t]
        Fintermediate = np.squeeze(np.sum(Fintermediate, axis=0))
        F[policy, t] = Fintermediate[-1]

    Gintermediate = np.zeros((NumPolicies, 1))
    horizon = T

    for policy in range(NumPolicies):
        if 'd' in MDP:
            for factor in range(NumFactors):
                Gintermediate[policy] -= d_complexity[factor + 1].T @ state_posterior[factor][:, 0, policy]
        for timestep in range(t, horizon):
            for factor in range(NumFactors):
                Expected_states[factor] = state_posterior[factor][:, timestep, policy]
            Gintermediate[policy] += G_epistemic_value(list(a.values()), list(Expected_states.values()))
            for modality in range(NumModalities):
                predictive_observations_posterior = cell_md_dot(a[modality + 1], Expected_states)
                Gintermediate[policy] += predictive_observations_posterior.T @ C[modality][:, t]
                if 'a' in MDP:
                    Gintermediate[policy] -= cell_md_dot(a_complexity[modality], [predictive_observations_posterior, *Expected_states])
    G[:, t] = Gintermediate.flatten()

    if t > 0:
        gamma[t] = gamma[t - 1]
    # For facilitation of calculating log(E) with different shape arrays in the iteration
    E = E.flatten()
    for ni in range(NumIterations):
        policy_priors[:, t] = np.exp(np.log(E) + gamma[t] * G[:, t]) / np.sum(np.exp(np.log(E) + gamma[t] * G[:, t]))
        policy_posteriors[:, t] = np.exp(np.log(E) + gamma[t] * G[:, t] + F[:, t]) / np.sum(np.exp(np.log(E) + gamma[t] * G[:, t] + F[:, t]))
        G_error = (policy_posteriors[:, t] - policy_priors[:, t]).T @ G[:, t]
        beta_update = posterior_beta - beta + G_error
        posterior_beta -= beta_update / 2
        gamma[t] = 1 / posterior_beta
        n = t * NumIterations + ni
        gamma_update[n, 0] = gamma[t].reshape(1, -1)
        policy_posterior_updates[:, n] = policy_posteriors[:, t]
        policy_posterior[:, t] = policy_posteriors[:, t]

    for factor in range(NumFactors):
        for tau in range(T):
            new_col = np.reshape(state_posterior[factor][:, tau, :], (NumStates[factor], NumPolicies)) @ policy_posteriors[:, t]
            new_col = new_col.reshape(1,-1).T
            if tau == 0:
                BMA_states[factor] = new_col
            else:
                BMA_states[factor] = np.hstack((BMA_states[factor], new_col))

    if t < T - 1:
        action_posterior_intermediate = np.zeros((NumControllable_transitions[-1], 1)).T
        for policy in range(NumPolicies):
            sub = tuple(V[t, policy, :].astype(int) - 1)
            action_posterior_intermediate[sub] += policy_posteriors[policy, t]
        # sub = (slice(None),) * NumFactors
        action_posterior_intermediate[:] = np.exp(alpha * np.log(action_posterior_intermediate[:])) / np.sum(np.exp(alpha * np.log(action_posterior_intermediate[:])))
        action_posterior[..., t] = action_posterior_intermediate
        ControlIndex = np.where(NumControllable_transitions > 1)[0]
        action = np.arange(1, NumControllable_transitions[ControlIndex] + 1)
        for factors in range(NumFactors):
            if NumControllable_transitions[factors] > 2:
                ind = np.argmax(np.cumsum(action_posterior_intermediate.flatten()) > np.random.rand())
                MDP['chosen_action'][factor, t] = action[ind]
                
                
# accumulate concentration paramaters (learning) --> MATLAB code L436

for t in range(T):
    # a matrix (likelihood)
    # but this part is never executed
    if 'a' in MDP:
        for modality in range(NumModalities):
            a_learning = O[modality, t].T
            for factor in range(NumFactors):
                a_learning = spm_cross(a_learning, BMA_states[factor][:, t])
            a_learning = a_learning * (MDP['a'][modality] > 0)
            MDP['a'][modality] = MDP['a'][modality] * omega + a_learning * eta

# Initial hidden states d (priors)
if 'd' in MDP:
    for factor in range(NumFactors):
        MDP['d'][factor + 1] = MDP['d'][factor + 1].astype(float)
        i = np.array(MDP['d'][factor + 1] > 0).flatten()
        if len(BMA_states[factor][i, 0]) == 1:
            MDP['d'][factor + 1][i] = omega * MDP['d'][factor + 1][i] + eta * BMA_states[factor][i, 0]
        else:
            MDP['d'][factor + 1][i] = omega * MDP['d'][factor + 1][i] + eta * BMA_states[factor][i, 0].reshape(MDP['d'][factor + 1].shape)
        
# Policies e (habits)
# but this part is never executed
if 'e' in MDP:
    MDP['e'] = omega * MDP['e'] + eta * policy_posteriors[:, T-1]

# Free energy of concentration parameters
# ----------------------------------------------------------------------

# (negative) free energy of a
# but this part is never executed
MDP['Fa'] = np.zeros(NumModalities)
for modality in range(1, NumModalities + 1):
    if 'a' in MDP:
        # Implement spm_KL_dir function for KL divergence calculation
        MDP['Fa'][modality-1] = - spm_KL_dir(MDP['a'][modality], a_prior[modality])

# (negative) free energy of d
MDP['Fd'] = np.zeros(NumFactors)
for factor in range(1, NumFactors + 1):
    if 'd' in MDP:
        MDP['Fd'][factor-1] = - spm_KL_dir(MDP['d'][factor], d_prior[factor])

# (negative) free energy of e
# but this part is never executed
if 'e' in MDP:
    MDP['Fe'] = - spm_KL_dir(MDP['e'], E)

# Simulated dopamine responses (beta updates)
# ----------------------------------------------------------------------
# "deconvolution" of neural encoding of precision
if NumPolicies > 1:
    # gamma_update = gamma  # Assuming gamma_update is defined in prior code
    phasic_dopamine = 8 * np.gradient(gamma_update.flatten()) + gamma_update.flatten() / 8
else:
    phasic_dopamine = []
    gamma_update = []

# Bayesian model average of neuronal variables; normalized firing rate and prediction error
# ----------------------------------------------------------------------
# Assuming Ni (NumIterations) is defined as 16 from prior code
Ni = NumIterations
BMA_normalized_firing_rates = {}
BMA_prediction_error = {}

for factor in range(NumFactors):
    num_states = NumStates[factor]  # NumStates is 0-indexed in Python
    BMA_normalized_firing_rates[factor + 1] = np.zeros((Ni, num_states, T, T))
    BMA_prediction_error[factor + 1] = np.zeros((Ni, num_states, T, T))
    
    for t in range(T):
        for policy in range(NumPolicies):    
            # Accumulate normalized firing rates
            BMA_normalized_firing_rates[factor + 1][:, :, :T, t] += (
                normalized_firing_rates[factor][:, :, :T, t, policy] * 
                policy_posteriors[policy, t]
            )
            
            # Accumulate prediction errors
            BMA_prediction_error[factor + 1][:, :, :T, t] += (
                prediction_error[factor][:, :, :T, t, policy] * 
                policy_posteriors[policy, t]
            )

print("Calculation completed. To be plotted.")