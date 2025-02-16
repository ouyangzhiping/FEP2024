import numpy as np
from spm_combination import spm_combinations
from spm_dir_norm import spm_dir_norm
from spm_MDP_size import spm_MDP_size
from spm_speye import spm_speye


def spm_MDP_check(MDP):
    """
    MDP structure checking
    """
    # deal with a sequence of trials
    if isinstance(MDP, list) and len(MDP) > 1:
        for m in range(len(MDP)):
            for i in range(len(MDP[m])):
                MDP[m][i] = spm_MDP_check(MDP[m][i])
        return MDP

    # fill in (posterior or process) likelihood and priors
    if 'A' not in MDP:
        MDP['A'] = MDP.get('a', None)
    if 'B' not in MDP:
        MDP['B'] = MDP.get('b', None)

    # check format of likelihood and priors
    if not isinstance(MDP['A'], list):
        MDP['A'] = [np.array(MDP['A'])]
    if not isinstance(MDP['B'], list):
        MDP['B'] = [np.array(MDP['B'])]

    if 'a' in MDP and not isinstance(MDP['a'], list):
        MDP['a'] = [np.array(MDP['a'])]
    if 'b' in MDP and not isinstance(MDP['b'], list):
        MDP['b'] = [np.array(MDP['b'])]

    # check dimensions and orders
    Nf = len(MDP['B'])  # number of hidden state factors
    NU = []
    NS = []
    for f in range(Nf):
        NU.append(MDP['B'][f].shape[2])  # number of hidden controls
        NS.append(MDP['B'][f].shape[0])  # number of hidden states
        MDP['B'][f] = MDP['B'][f].astype(float)
        MDP['B'][f] = spm_dir_norm(MDP['B'][f])

    Ng = len(MDP['A'])  # number of outcome factors
    No = []
    for g in range(Ng):
        No.append(MDP['A'][g].shape[0])  # number of outcomes
        if not (np.issparse(MDP['A'][g]) or np.islogical(MDP['A'][g])):
            MDP['A'][g] = MDP['A'][g].astype(float)
        if not np.islogical(MDP['A'][g]):
            MDP['A'][g] = spm_dir_norm(MDP['A'][g])

    # check sizes of Dirichlet parameterisation
    Nf, Ns, Nu = spm_MDP_size(MDP)

    # check policy specification (create default moving policy U, if necessary)
    if 'U' in MDP:
        if MDP['U'].shape[0] == 1 and MDP['U'].shape[2] == Nf:
            MDP['U'] = np.moveaxis(MDP['U'], 0, -1)
    try:
        V = np.expand_dims(MDP['U'], axis=0)
    except:
        try:
            V = MDP['V']
        except:
            MDP['U'] = spm_combinations(Nu)
            V = np.expand_dims(MDP['U'], axis=0)
    MDP['V'] = V

    # check policy specification
    if Nf != V.shape[2] and V.shape[2] > 1:
        raise ValueError('please ensure V[:,:,1:Nf] is consistent with MDP.B{1:Nf}')

    # check preferences
    if 'C' not in MDP:
        MDP['C'] = [np.zeros((No[g], 1)) for g in range(Ng)]
    for g in range(Ng):
        if isinstance(MDP['C'], list):
            if MDP['C'][g].ndim == 1:
                MDP['C'][g] = MDP['C'][g].reshape(-1, 1)
            if No[g] != MDP['C'][g].shape[0]:
                raise ValueError(f'please ensure A[{g}] and C[{g}] are consistent')

    # check initial states
    if 'D' not in MDP:
        MDP['D'] = [np.ones((Ns[f], 1)) for f in range(Nf)]
    if Nf != len(MDP['D']):
        raise ValueError('please check MDP.D')
    for f in range(Nf):
        MDP['D'][f] = MDP['D'][f].reshape(-1, 1)

    # check initial controls
    # if 'E' not in MDP:
    #     MDP['E'] = [np.ones((Nu[f], 1)) for f in range(Nf)]
    # if Nf != len(MDP['E']):
    #     raise ValueError('please check MDP.E')
    # for f in range(Nf):
    #     MDP['E'][f] = MDP['E'][f].reshape(-1, 1)

    # check initial states and internal consistency
    for f in range(Nf):
        if Ns[f] != MDP['D'][f].shape[0]:
            raise ValueError(f'please ensure B[{f}] and D[{f}] are consistent')
        if V.shape[2] > 1:
            if Nu[f] < np.max(V[:, :, f]):
                raise ValueError(f'please check V[:,:,{f}] or U[:,:,{f}]')
        for g in range(Ng):
            try:
                Na = MDP['a'][g].shape
            except:
                Na = MDP['A'][g].shape
            if not all(Na[1:] == Ns):
                raise ValueError(f'please ensure A[{g}] and D[{f}] are consistent')

    # check probability matrices are properly specified
    for f in range(len(MDP['B'])):
        if not np.all(np.any(MDP['B'][f], axis=0)):
            raise ValueError(f'please check B[{f}] for missing entries')
    for g in range(len(MDP['A'])):
        if not np.all(np.any(MDP['A'][g], axis=0)):
            raise ValueError(f'please check A[{g}] for missing entries')

    # check initial states
    if 's' in MDP:
        if MDP['s'].shape[0] > len(MDP['B']):
            raise ValueError(f'please specify an initial state MDP.s for {Nf} factors')
        f = np.max(MDP['s'], axis=1)
        if np.any(f > NS[:len(f)]):
            raise ValueError('please ensure initial states MDP.s are consistent with MDP.B')

    # check outcomes if specified
    if 'o' in MDP:
        if len(MDP['o']):
            if MDP['o'].shape[0] != Ng:
                raise ValueError(f'please specify an outcomes MDP.o for {Ng} modalities')
            if np.any(np.max(MDP['o'], axis=1) > No):
                raise ValueError('please ensure # outcomes MDP.o are consistent with MDP.A')

    # check (primary link array if necessary)
    if 'link' in MDP:
        nf = len(MDP['MDP'][0]['B'])
        ns = [MDP['MDP'][0]['B'][f].shape[0] for f in range(nf)]
        if not all(np.array(MDP['link']).shape == [nf, Ng]):
            raise ValueError(f'please check the size of link [{nf},{Ng}]')
        if isinstance(MDP['link'], np.ndarray):
            link = [[None for _ in range(Ng)] for _ in range(nf)]
            for f in range(len(MDP['link'])):
                for g in range(len(MDP['link'][f])):
                    if MDP['link'][f][g]:
                        link[f][g] = spm_speye(ns[f], No[g], 0)
            MDP['link'] = link
        for f in range(len(MDP['link'])):
            for g in range(len(MDP['link'][f])):
                if MDP['link'][f][g] is not None:
                    if not all(np.array(MDP['link'][f][g]).shape == [ns[f], No[g]]):
                        raise ValueError(f'please check link[{f},{g}]')

    # Empirical prior preferences
    if 'linkC' in MDP:
        if isinstance(MDP['linkC'], np.ndarray):
            linkC = [[None for _ in range(Ng)] for _ in range(len(MDP['MDP']['C']))]
            for f in range(len(MDP['linkC'])):
                for g in range(len(MDP['linkC'][f])):
                    if MDP['linkC'][f][g]:
                        linkC[f][g] = spm_speye(MDP['MDP']['C'][f].shape[0], No[g], 0)
            MDP['linkC'] = linkC

    # Empirical priors over policies
    if 'linkE' in MDP:
        if isinstance(MDP['linkE'], np.ndarray):
            linkE = [None for _ in range(Ng)]
            for g in range(len(MDP['linkE'][0])):
                if MDP['linkE'][0][g]:
                    linkE[g] = spm_speye(MDP['MDP']['E'].shape[0], No[g], 0)
            MDP['linkE'] = linkE

    # check factors and outcome modalities have proper labels
    for i in range(Nf):
        try:
            MDP['label']['factor'][i]
        except:
            try:
                MDP['label']['factor'][i] = MDP['Bname'][i]
            except:
                MDP['label']['factor'][i] = f'factor {i}'
        for j in range(NS[i]):
            try:
                MDP['label']['name'][i][j]
            except:
                try:
                    MDP['label']['name'][i][j] = MDP['Sname'][i][j]
                except:
                    MDP['label']['name'][i][j] = f'state {j}({i})'
        for j in range(Nu[i]):
            try:
                MDP['label']['action'][i][j]
            except:
                MDP['label']['action'][i][j] = f'act {j}({i})'

    for i in range(Ng):
        try:
            MDP['label']['modality'][i]
        except:
            try:
                MDP['label']['modality'][i] = MDP['Bname'][i]
            except:
                MDP['label']['modality'][i] = f'modality {i}'
        for j in range(No[i]):
            try:
                MDP['label']['outcome'][i][j]
            except:
                try:
                    MDP['label']['outcome'][i][j] = MDP['Oname'][i][j]
                except:
                    MDP['label']['outcome'][i][j] = f'outcome {j}({i})'

    # check names are specified properly
    if 'Aname' in MDP:
        if len(MDP['Aname']) != Ng:
            raise ValueError('please specify an MDP.Aname for each modality')
    if 'Bname' in MDP:
        if len(MDP['Bname']) != Nf:
            raise ValueError('please specify an MDP.Bname for each factor')

    return MDP
    
