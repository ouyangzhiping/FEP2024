import numpy as np

###############################################################################

# 参考《主动推理》“参数估测”：P257

def Estimate_parameters(DCM):
    """
    MDP inversion using Variational Bayes
    FORMAT [DCM] = spm_dcm_mdp(DCM)
    
    Expects:
    --------------------------------------------------------------------------
    DCM.MDP   # MDP structure specifying a generative model
    DCM.field # parameter (field) names to optimise
    DCM.U     # list of outcomes (stimuli)
    DCM.Y     # list of responses (action)
    
    Returns:
    --------------------------------------------------------------------------
    DCM.M     # generative model (DCM)
    DCM.Ep    # Conditional means (structure)
    DCM.Cp    # Conditional covariances
    DCM.F     # (negative) Free-energy bound on log evidence
    
    This routine inverts (list of) trials specified in terms of the
    stimuli or outcomes and subsequent choices or responses. It first
    computes the prior expectations (and covariances) of the free parameters
    specified by DCM.field. These parameters are log scaling parameters that
    are applied to the fields of DCM.MDP. 
    
    If there is no learning implicit in multi-trial games, only unique trials
    (as specified by the stimuli), are used to generate (subjective)
    posteriors over choice or action. Otherwise, all trials are used in the
    order specified. The ensuing posterior probabilities over choices are
    used with the specified choices or actions to evaluate their log
    probability. This is used to optimise the MDP (hyper) parameters in
    DCM.field using variational Laplace (with numerical evaluation of the
    curvature).
    """
    
    # OPTIONS
    ALL = False

    # Here we specify prior expectations (for parameter means and variances)
    prior_variance = 1/4  # smaller values will lead to a greater complexity 
                          # penalty (posteriors will remain closer to priors)

    pE = {}
    pC = {}

    for i, field in enumerate(DCM['field']):
        try:
            param = DCM['MDP'][field]
            param = np.double(param != 0)
        except KeyError:
            param = 1
        if ALL:
            pE[field] = np.zeros_like(param)
            pC[(i, i)] = np.diag(param)
        else:
            if field == 'alpha':
                pE[field] = np.log(16)          # in log-space (to keep positive)
                pC[(i, i)] = prior_variance
            elif field == 'beta':
                pE[field] = np.log(1)           # in log-space (to keep positive)
                pC[(i, i)] = prior_variance
            elif field == 'la':
                pE[field] = np.log(1)           # in log-space (to keep positive)
                pC[(i, i)] = prior_variance
            elif field == 'rs':
                pE[field] = np.log(5)           # in log-space (to keep positive)
                pC[(i, i)] = prior_variance
            elif field == 'eta':
                pE[field] = np.log(0.5 / (1 - 0.5))  # in logit-space - bounded between 0 and 1
                pC[(i, i)] = prior_variance
            elif field == 'omega':
                pE[field] = np.log(0.5 / (1 - 0.5))  # in logit-space - bounded between 0 and 1
                pC[(i, i)] = prior_variance
            else:
                pE[field] = 0                # if it can take any negative or positive value
                pC[(i, i)] = prior_variance

    pC = spm_cat(pC)

    # model specification
    M = {
        'L': lambda P, M, U, Y: spm_mdp_L(P, M, U, Y),  # log-likelihood function
        'pE': pE,                                      # prior means (parameters)
        'pC': pC,                                      # prior variance (parameters)
        'mdp': DCM['MDP']                              # MDP structure
    }

    # Variational Laplace
    Ep, Cp, F = spm_nlsi_Newton(M, DCM['U'], DCM['Y'])  # This is the actual fitting routine

    # Store posterior distributions and log evidence (free energy)
    DCM['M'] = M  # Generative model
    DCM['Ep'] = Ep  # Posterior parameter estimates
    DCM['Cp'] = Cp  # Posterior variances and covariances
    DCM['F'] = F  # Free energy of model fit

    return DCM

def spm_mdp_L(P, M, U, Y):
    """
    log-likelihood function
    FORMAT L = spm_mdp_L(P,M,U,Y)
    P    - parameter structure
    M    - generative model
    U    - inputs
    Y    - observed responses
    
    This function runs the generative model with a given set of parameter
    values, after adding in the observations and actions on each trial
    from (real or simulated) participant data. It then sums the
    (log-)probabilities (log-likelihood) of the participant's actions under the model when it
    includes that set of parameter values. The variational Bayes fitting
    routine above uses this function to find the set of parameter values that maximize
    the probability of the participant's actions under the model (while also
    penalizing models with parameter values that move farther away from prior
    values).
    """
    
    if not isinstance(P, dict):
        P = spm_unvec(P, M['pE'])

    # Here we re-transform parameter values out of log- or logit-space when 
    # inserting them into the model to compute the log-likelihood
    mdp = M['mdp']
    fields = M['pE'].keys()
    for field in fields:
        if field == 'alpha':
            mdp[field] = np.exp(P[field])
        elif field == 'beta':
            mdp[field] = np.exp(P[field])
        elif field == 'la':
            mdp[field] = np.exp(P[field])
        elif field == 'rs':
            mdp[field] = np.exp(P[field])
        elif field == 'eta':
            mdp[field] = 1 / (1 + np.exp(-P[field]))
        elif field == 'omega':
            mdp[field] = 1 / (1 + np.exp(-P[field]))
        else:
            mdp[field] = np.exp(P[field])

    # place MDP in trial structure
    la = mdp['la_true']  # true level of loss aversion
    rs = mdp['rs_true']  # true preference magnitude for winning (higher = more risk-seeking)

    if 'la' in M['pE'] and 'rs' in M['pE']:
        mdp['C'][2] = np.array([[0, 0, 0],      # Null
                                [0, -mdp['la'], -mdp['la']],  # Loss
                                [0, mdp['rs'], mdp['rs'] / 2]])  # win
    elif 'la' in M['pE']:
        mdp['C'][2] = np.array([[0, 0, 0],      # Null
                                [0, -mdp['la'], -mdp['la']],  # Loss
                                [0, rs, rs / 2]])  # win
    elif 'rs' in M['pE']:
        mdp['C'][2] = np.array([[0, 0, 0],      # Null
                                [0, -la, -la],  # Loss
                                [0, mdp['rs'], mdp['rs'] / 2]])  # win
    else:
        mdp['C'][2] = np.array([[0, 0, 0],  # Null
                                [0, -la, -la],  # Loss
                                [0, rs, rs / 2]])  # win

    j = range(len(U))  # observations for each trial
    n = len(j)  # number of trials

    MDP = [mdp] * n  # Create MDP with number of specified trials
    for k in j:
        MDP[k]['o'] = U[k]  # Add observations in each trial

    # solve MDP and accumulate log-likelihood
    MDP = spm_MDP_VB_X_tutorial(MDP)  # run model with possible parameter values

    L = 0  # start (log) probability of actions given the model at 0

    for i in range(len(Y)):  # Get probability of true actions for each trial
        for j in range(len(Y[0][1])):  # Only get probability of the second (controllable) state factor
            L += np.log(MDP[i]['P'][:, Y[i][1][j], j] + np.finfo(float).eps)  # sum the (log) probabilities of each action
                                                                              # given a set of possible parameter values

    print(f'LL: {L}')
    return L

# def spm_cat(pC):
#     # This function concatenates the covariance matrices
#     # Placeholder implementation
#     LEN_KEY_I = len(set([key[0] for key in pC.keys()]))
#     LEN_KEY_J = len(set([key[1] for key in pC.keys()]))
#     return np.block([[pC.get((i, j), np.zeros((1, 1))) for j in range(LEN_KEY_J)] for i in range(LEN_KEY_I)])

def spm_nlsi_Newton(M, U, Y):
    # Placeholder implementation for the variational Laplace fitting routine
    # This should be replaced with the actual implementation
    Ep = M['pE']
    Cp = M['pC']
    F = -np.inf  # Free energy (log evidence)
    return Ep, Cp, F

def spm_unvec(P, pE):
    # This function converts a vector back to a structure
    # Placeholder implementation
    return pE

def spm_MDP_VB_X_tutorial(MDP):
    # Placeholder implementation for the MDP solver
    # This should be replaced with the actual implementation
    for mdp in MDP:
        mdp['P'] = np.random.rand(3, 3, 3)  # Random probabilities for demonstration
    return MDP