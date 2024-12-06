def spm_cat(x):
    result = {}
    for (i, j), values in x.items():
        for k, value in enumerate(values):
            result[(j, k + i * len(values))] = [value]
    return result