import numpy as np

def fp_to_fxp(x, m, n, saturate=True, rounding='nearest'):
    total_bits = m + n + 1
    scale = 1 << n

    # Compute fixed-point raw value
    if rounding == 'nearest':
        x_scaled = np.round(x * scale)
    elif rounding == 'floor':
        x_scaled = np.floor(x * scale)
    else:
        raise ValueError("rounding must be 'nearest' or 'floor'")

    #print(x_scaled)

    min_val = -(1 << (m + n))
    max_val = (1 << (m + n)) - 1

    # Saturate if needed
    if saturate:
        x_scaled = np.clip(x_scaled, min_val, max_val)

    return x_scaled.astype(np.int64) 

def cfp_to_cfxp(x, m, n, saturate=True, rounding='nearest'):
    x_real = np.real(x)
    x_imag = np.imag(x)

    x_real_scaled = fp_to_fxp(x_real, m, n, saturate, rounding)
    x_imag_scaled = fp_to_fxp(x_imag, m, n, saturate, rounding)

    x_complex_scaled = x_real_scaled + 1j * x_imag_scaled

    return x_complex_scaled 

def gen_16QAM_symbol_vec(Nt):
    const = np.array([-3, -1, 1, 3]) / np.sqrt(10)

    real_part = np.random.choice(const, size=(Nt, 1))
    imag_part = np.random.choice(const, size=(Nt, 1))

    X = real_part + 1j * imag_part  # Shape: (n_samples, Nt)

    return X


def to_int16(x, mode="saturate"):
    INT16_MIN, INT16_MAX = -32768, 32767

    if mode == "saturate":
        # 饱和
        if x > INT16_MAX:
            return INT16_MAX
        elif x < INT16_MIN:
            return INT16_MIN
        else:
            return int(x)

    elif mode == "wrap":
        # wrap-around (补码截断)
        return np.array([x], dtype=np.int16)[0]

    elif mode == "error":
        # 严格模式
        return np.int16(x)  # 超范围会触发 OverflowError

    else:
        raise ValueError("mode must be 'saturate', 'wrap', or 'error'")

def to_int(x, wl, mode="saturate"):
    assert wl in (16, 32)
    if wl == 16:
        return to_int16(x, mode)
