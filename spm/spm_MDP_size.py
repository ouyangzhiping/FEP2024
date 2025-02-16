def spm_MDP_size(mdp):
    """
    Dimensions of MDP
    :param mdp: dictionary containing MDP parameters
    :return: tuple (Nf, Ns, Nu, Ng, No)
    Nf  - number of factors
    Ns  - states per factor
    Nu  - control per factors
    Ng  - number of modalities
    No  - levels per modality
    """
    
    # checks
    if 'a' not in mdp:
        mdp['a'] = mdp['A']
    if 'b' not in mdp:
        mdp['b'] = mdp['B']
    
    # sizes of factors and modalities
    Nf = len(mdp['b'])  # number of hidden factors
    Ng = len(mdp['a'])  # number of outcome modalities
    Ns = [0] * Nf
    Nu = [0] * Nf
    No = [0] * Ng
    
    for f in range(Nf):
        Ns[f] = mdp['b'][f].shape[0]  # number of hidden states
        Nu[f] = mdp['b'][f].shape[2]  # number of hidden controls
    
    for g in range(Ng):
        No[g] = mdp['a'][g].shape[0]  # number of outcomes
    
    return Nf, Ns, Nu, Ng, No