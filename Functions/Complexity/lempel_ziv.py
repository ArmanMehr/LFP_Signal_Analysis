import numpy as np

def lempel_ziv(signal, num_levels = 2):
    med = np.median(signal)

    if np.isnan(signal[0]):
        return np.nan, None

    if num_levels == 2:
        P = (np.sign(signal - med) + 1) / 2
        P[signal == med] = 0

    elif num_levels == 3:
        P = np.zeros_like(signal)
        P[signal >= med + abs(max(signal)) / 16] = 2
        P[(signal > med - abs(max(signal)) / 16) & (signal < med + abs(max(signal)) / 16)] = 1

    c = 2
    terminate = False
    r = 1
    i = 1

    while not terminate:
        S = P[:r]
        Q = P[r:r+i]
        concat = np.concatenate((S, Q))

        if not belong_to_voc2(Q, concat[:len(concat)-1]):
            c += 1
            r += i
            i = 1
        else:
            i += 1

        if r + i == len(P):
            terminate = True

    out = c * np.log2(len(P)) / len(P)
    if num_levels == 3:
        out = out / np.log2(3)

    return out, P

def belong_to_voc2(string1, string2):
    size_string1 = len(string1)
    size_string2 = len(string2)

    for i in range(size_string2 - size_string1 + 1):
        interval_string = string2[i:i + size_string1]
        if np.array_equal(interval_string, string1):
            return True
    return False