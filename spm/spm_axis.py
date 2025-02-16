import matplotlib.pyplot as plt
import numpy as np

def spm_axis(*args):
    if len(args) == 0:
        raise ValueError("No arguments provided")

    if len(args) == 1 and args[0] in ['tight', 'scale']:
        spm_axis(plt.gca(), args[0])
    elif len(args) == 2 and all_axes(args[0]) and args[1] == 'tight':
        for ax in args[0]:
            ylim = ax.get_ylim()
            if np.diff(ylim) < 1e-12:
                ax.set_ylim(ylim[0] - 1, ylim[1] + 1)
            else:
                ax.set_ylim(ylim[0] - np.diff(ylim) / 16, ylim[1] + np.diff(ylim) / 16)
    elif len(args) == 2 and all_axes(args[0]) and args[1] == 'scale':
        for ax in args[0]:
            ylim = ax.get_ylim()
            ax.set_ylim(0, ylim[1] * (1 + 1/16))
    else:
        plt.axis(*args)

def all_axes(handles):
    return all(isinstance(h, plt.Axes) for h in handles) and len(handles) == len([h for h in handles if isinstance(h, plt.Axes)])