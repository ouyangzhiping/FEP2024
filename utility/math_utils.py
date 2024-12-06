import numpy as np

def nat_log(x):
    return np.log(x + np.exp(-16))

def dic2mat(dic):
    # 获取矩阵的最大行和列
    max_row = max(row for row, col in dic.keys()) + 1
    max_col = max(col for row, col in dic.keys()) + 1
    
    # 初始化全矩阵
    matrix = np.zeros((max_row, max_col), dtype=float)
    
    # 将矩阵的值放置在全矩阵的正确位置
    for (row, col), value in dic.items():
        if isinstance(value[0], list):
            matrix[row, col] = float(value[0][0])
        else:
            matrix[row, col] = float(value[0])

    return matrix