import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

def spm_MDP_VB_LFP(MDP, UNITS=None, f=1, SPECTRAL=0):
    # 检查是否有模拟的神经元响应
    if 'xn' not in MDP[0]:
        print('请使用其他反演方案来模拟神经元响应（例如，spm_MDP_VB_XX）')
        return

    # 默认值
    if UNITS is None:
        UNITS = []

    # 维度
    Nt = len(MDP)  # 试验次数
    try:
        Ne = MDP[0]['xn'][f].shape[3]  # 纪元数
        Nx = MDP[0]['B'][f].shape[0]  # 状态数
        Nb = MDP[0]['xn'][f].shape[0]  # 每个纪元的时间段数
    except:
        Ne = MDP[0]['xn'].shape[3]
        Nx = MDP[0]['A'].shape[1]
        Nb = MDP[0]['xn'].shape[0]

    # 要绘制的单元
    ALL = []
    for i in range(Ne):
        for j in range(Nx):
            ALL.append([j, i])
    if len(ALL) > 512:
        ii = np.round(np.linspace(0, len(ALL) - 1, 512)).astype(int)
        ALL = [ALL[i] for i in ii]
    if not UNITS:
        UNITS = ALL
    ii = list(range(len(ALL)))

    # 汇总统计：发射率
    z = []
    v = []
    dn = []
    for i in range(Nt):
        str_list = []
        try:
            xn = MDP[i]['xn'][f]
        except:
            xn = MDP[i]['xn']
        z.append([[xn[:, ALL[j][0], ALL[j][1], k] for k in range(Ne)] for j in range(len(ALL))])
        v.append([[xn[:, UNITS[j][0], UNITS[j][1], k] for k in range(Ne)] for j in range(len(UNITS))])
        dn.append(np.mean(MDP[i]['dn'], axis=1))

    if len(dn) == 0:
        return

    # 相位幅度耦合
    dt = 1 / 64  # 时间段（秒）
    t = np.arange(1, Nb * Ne * Nt + 1) * dt  # 时间（秒）
    Hz = np.arange(4, 33)  # 频率范围
    n = 1 / (4 * dt)  # 窗口长度
    w = Hz * (dt * n)  # 每个窗口的周期数

    # 模拟发射率
    z = np.concatenate([np.concatenate(z[i], axis=1) for i in range(len(z))], axis=1).T
    v = np.concatenate([np.concatenate(v[i], axis=1) for i in range(len(v))], axis=1).T

    # 带通滤波器在 8 到 32 Hz 之间的对数率：局部场电位
    c = 1 / 32
    x = np.log(z.T + c)
    u = np.log(v.T + c)
    x = convolve(x, np.ones((2,)) / 2, mode='same') - convolve(x, np.ones((16,)) / 16, mode='same')
    u = convolve(u, np.ones((2,)) / 2, mode='same') - convolve(u, np.ones((16,)) / 16, mode='same')

    # 绘图
    fig, axs = plt.subplots(4, 1, figsize=(10, 15))
    axs[0].imshow(64 * (1 - z), aspect='auto', extent=[t[0], t[-1], 0, len(ii)])
    axs[0].set_title(MDP[0]['label']['factor'][f])
    axs[0].set_xlabel('time (sec)')
    if len(str_list) < 16:
        axs[0].grid(True)
        axs[0].set_yticks(range(Ne * Nx))
        axs[0].set_yticklabels(str_list)
    axs[0].grid(True)
    axs[0].set_xticks(np.arange(1, Ne * Nt + 1) * Nb * dt)
    if Ne * Nt > 32:
        axs[0].set_xticklabels([])
    if Nt == 1:
        axs[0].axis('square')

    # 时间频率分析和 theta 相位
    wft = np.abs(np.fft.fft(x, n=int(n), axis=0))
    csd = np.sum(wft, axis=2)
    lfp = np.sum(x, axis=1)
    phi = np.angle(np.fft.ifft(np.sum(wft[0, :, :], axis=2), n=int(n)))
    lfp = 4 * lfp / np.std(lfp) + 16
    phi = 4 * phi / np.std(phi) + 16

    axs[1].imshow(csd, aspect='auto', extent=[t[0], t[-1], Hz[0], Hz[-1]], origin='lower')
    axs[1].plot(t, lfp, 'w:', t, phi, 'w')
    axs[1].grid(True)
    axs[1].set_xticks(np.arange(1, Ne * Nt + 1) * Nb * dt)
    axs[1].set_title('Time-frequency response')
    axs[1].set_xlabel('time (sec)')
    axs[1].set_ylabel('frequency (Hz)')
    if Nt == 1:
        axs[1].axis('square')

    # 频谱响应
    if SPECTRAL:
        fig, axs = plt.subplots(4, 2, figsize=(10, 15))
        csd = np.sum(np.abs(wft), axis=1)
        axs[0, 0].plot(Hz, np.log(csd))
        axs[0, 0].set_title('Spectral response')
        axs[0, 0].set_xlabel('frequency (Hz)')
        axs[0, 0].set_ylabel('log power')
        axs[0, 0].axis('tight')
        axs[0, 0].box(False)
        axs[0, 0].axis('square')

        cfc = 0
        for i in range(wft.shape[2]):
            cfc += np.corrcoef(np.abs(wft[:, :, i]).T)
        axs[0, 1].imshow(cfc, aspect='auto', extent=[Hz[0], Hz[-1], Hz[0], Hz[-1]], origin='lower')
        axs[0, 1].set_title('Cross-frequency coupling')
        axs[0, 1].set_xlabel('frequency (Hz)')
        axs[0, 1].set_ylabel('frequency (Hz)')
        axs[0, 1].box(False)
        axs[0, 1].axis('square')

    # 局部场电位
    axs[2].plot(t, u)
    axs[2].plot(t, x, ':')
    axs[2].grid(True)
    axs[2].set_xticks(np.arange(1, Ne * Nt + 1) * Nb * dt)
    for i in range(2, Nt + 1, 2):
        axs[2].axvspan((i - 1) * Ne * Nb * dt, i * Ne * Nb * dt, color='w', alpha=0.1)
    axs[2].set_title('Local field potentials')
    axs[2].set_xlabel('time (sec)')
    axs[2].set_ylabel('response')
    if Nt == 1:
        axs[2].axis('square')
    axs[2].box(False)

    # 发射率
    if Nt == 1:
        axs[3].plot(t, v)
        axs[3].plot(t, z, ':')
        axs[3].grid(True)
        axs[3].set_xticks(np.arange(1, Ne * Nt + 1) * Nb * dt)
        axs[3].set_title('Firing rates')
        axs[3].set_xlabel('time (sec)')
        axs[3].set_ylabel('response')
        axs[3].axis('square')

    # 模拟多巴胺响应（如果不是移动策略）
    dn = np.concatenate(dn)
    dn = dn * (dn > 0)
    dn = dn + (dn + 1 / 16) * np.random.rand(len(dn)) / 8
    axs[3].bar(np.arange(len(dn)), dn, color='k')
    axs[3].set_title('Dopamine responses')
    axs[3].set_xlabel('time (updates)')
    axs[3].set_ylabel('change in precision')
    axs[3].axis('tight')
    axs[3].box(False)
    axs[3].set_ylim(bottom=0)
    if Nt == 1:
        axs[3].axis('square')

    # 模拟光栅
    if Nt == 1 and len(ii) < 129:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        R = np.kron(z, np.ones((16, 16)))
        R = np.random.rand(*R.shape) > R * (1 - 1 / 16)
        ax.imshow(R, aspect='auto', extent=[t[0], t[-1], 0, Nx * Ne])
        ax.set_title('Unit firing')
        ax.set_xlabel('time (sec)')
        ax.grid(True)
        ax.set_xticks(np.arange(1, Ne * Nt + 1) * Nb * dt)
        ax.set_yticks(range(Ne * Nx))
        ax.set_yticklabels(str_list)
        ax.axis('square')

    plt.show()

# 示例调用
# MDP = [{'xn': ..., 'dn': ..., 'label': {'factor': ..., 'name': ...}, 'B': ..., 'A': ...}]
# spm_MDP_VB_LFP(MDP)