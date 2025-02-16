import numpy as np
import matplotlib.pyplot as plt
from spm_MDP_check import spm_MDP_check

def spm_MDP_VB_trial(MDP, gf=None, gg=None):
    """
    Auxiliary plotting routine for spm_MDP_VB - single trial
    """
    # Check MDP structure
    MDP = spm_MDP_check(MDP)
    plt.clf()

    # Number of transitions, policies and states
    if isinstance(MDP['X'], list):
        Nf = len(MDP['B'])  # number of hidden state factors
        Ng = len(MDP['A'])  # number of outcome factors
        X = MDP['X']
        C = MDP['C']
        Nu = [np.size(MDP['B'][f], 2) > 1 for f in range(Nf)]
    else:
        Nf = 1
        Ng = 1
        Nu = [1]
        X = [MDP['X']]
        C = [MDP['C']]

    # Factors and outcomes to plot
    maxg = 3
    if gf is None:
        gf = list(range(1, min(Nf, maxg) + 1))
    if gg is None:
        gg = list(range(1, min(Ng, maxg) + 1))
    nf = len(gf)
    ng = len(gg)

    # Posterior beliefs about hidden states
    for f in range(nf):
        plt.subplot(3 * nf, 2, (f) * 2 + 1)
        plt.imshow(64 * (1 - X[gf[f] - 1]), cmap='gray')
        if X[gf[f] - 1].shape[0] > 128:
            spm_spy(X[gf[f] - 1], 12, 1)
        a = plt.axis()
        if 's' in MDP:
            plt.plot(MDP['s'][gf[f] - 1, :], '.r', markersize=8)
            plt.axis(a)
        if f < 1:
            plt.title(f'Hidden states - {MDP["label"]["factor"][gf[f] - 1]}')
        else:
            plt.title(MDP["label"]["factor"][gf[f] - 1])
        plt.gca().set_xticklabels([])
        plt.gca().set_xticks(range(1, X[0].shape[1] + 1))

        YTickLabel = MDP["label"]["name"][gf[f] - 1]
        if len(YTickLabel) > 8:
            i = np.linspace(1, len(YTickLabel), 8)
            YTickLabel = [YTickLabel[int(round(idx)) - 1] for idx in i]
        else:
            i = range(1, len(YTickLabel) + 1)
        plt.gca().set_yticks(i)
        plt.gca().set_yticklabels(YTickLabel)

    # Posterior beliefs about control states
    Nu = [i for i, val in enumerate(Nu) if val]
    Np = len(Nu)
    for f in range(Np):
        plt.subplot(3 * Np, 2, (f + 1) * 2)
        if isinstance(MDP['P'], list):
            P = MDP['P'][f]
        elif Nf > 1:
            ind = list(range(1, Nf + 1))
            P = MDP['P']
            for dim in range(Nf):
                if dim != ind[Nu[f]]:
                    P = np.sum(P, axis=dim)
            P = np.squeeze(P)
        else:
            P = np.squeeze(MDP['P'])

        # Display
        plt.imshow(64 * (1 - P), cmap='gray')
        if 'u' in MDP:
            plt.plot(MDP['u'][Nu[f], :], '.c', markersize=16)
        if f < 1:
            plt.title(f'Action - {MDP["label"]["factor"][Nu[f]]}')
        else:
            plt.title(MDP["label"]["factor"][Nu[f]])
        plt.gca().set_xticklabels([])
        plt.gca().set_xticks(range(1, X[0].shape[1] + 1))

        YTickLabel = MDP["label"]["action"][Nu[f]]
        if len(YTickLabel) > 8:
            i = np.round(np.linspace(1, len(YTickLabel), 8))
            YTickLabel = [YTickLabel[int(idx) - 1] for idx in i]
        else:
            i = range(1, len(YTickLabel) + 1)
        plt.gca().set_yticks(i)
        plt.gca().set_yticklabels(YTickLabel)

        # Policies
        plt.subplot(3 * Np, 2, (Np + f) * 2 + 1)
        plt.imshow(MDP['V'][:, :, Nu[f]].T, cmap='gray')
        if f < 1:
            plt.title(f'Allowable policies - {MDP["label"]["factor"][Nu[f]]}')
        else:
            plt.title(MDP["label"]["factor"][Nu[f]])
        if f < Np - 1:
            plt.gca().set_xticklabels([])
        plt.gca().set_xticks(range(1, X[0].shape[1]))

    # Expectations over policies
    if 'un' in MDP:
        plt.subplot(3, 2, 4)
        plt.imshow(64 * (1 - MDP['un']), cmap='gray')
        plt.title('Posterior probability')
        plt.ylabel('policy')
        plt.xlabel('updates')

    # Sample (observation) and preferences
    for g in range(ng):
        plt.subplot(3 * ng, 2, (2 * ng + g) * 2 + 1)
        c = C[gg[g] - 1]
        if c.shape[1] < MDP['o'].shape[1]:
            c = np.tile(c[:, 0], (1, MDP['o'].shape[1]))
        if c.shape[0] > 128:
            spm_spy(c, 16, 1)
        else:
            plt.imshow(1 - c, cmap='gray')
        plt.plot(MDP['o'][gg[g] - 1, :], '.c', markersize=16)
        if g < 1:
            plt.title(f'Outcomes and preferences - {MDP["label"]["modality"][gg[g] - 1]}')
        else:
            plt.title(MDP["label"]["modality"][gg[g] - 1])
        if g == ng - 1:
            plt.xlabel('time')
        else:
            plt.gca().set_xticklabels([])
        plt.gca().set_xticks(range(1, X[0].shape[1]))

        YTickLabel = MDP["label"]["outcome"][gg[g] - 1]
        if len(YTickLabel) > 8:
            i = np.round(np.linspace(1, len(YTickLabel), 8))
            YTickLabel = [YTickLabel[int(idx) - 1] for idx in i]
        else:
            i = range(1, len(YTickLabel) + 1)
        plt.gca().set_yticks(i)
        plt.gca().set_yticklabels(YTickLabel)

    # Expected precision
    if 'dn' in MDP and 'wn' in MDP:
        if MDP['dn'].shape[1] > 0:
            plt.subplot(3, 2, 6)
            if MDP['dn'].shape[1] > 1:
                plt.plot(MDP['dn'], 'r:')
                plt.plot(MDP['wn'], 'c', linewidth=2)
            else:
                plt.bar(range(len(MDP['dn'])), MDP['dn'], 1.1, color='k')
                plt.plot(MDP['wn'], 'c', linewidth=2)
            plt.title('Expected precision (dopamine)')
            plt.xlabel('updates')
            plt.ylabel('precision')
            plt.tight_layout()
            plt.box(False)
    plt.draw()

def spm_spy(matrix, threshold, size):
    # This function should visualize the matrix
    # Placeholder for the actual implementation
    pass