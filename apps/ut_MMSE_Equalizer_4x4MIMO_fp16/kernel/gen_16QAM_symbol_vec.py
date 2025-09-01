import numpy as np

def gen_16QAM_symbol_vec(Nt):
    const = np.array([-3, -1, 1, 3]) / np.sqrt(10)

    real_part = np.random.choice(const, size=(Nt, 1))
    imag_part = np.random.choice(const, size=(Nt, 1))

    X = real_part + 1j * imag_part  # Shape: (n_samples, Nt)

    return X

if __name__ == "__main__":
    X_ref_real = np.zeros((4, 1, 16))
    X_ref_imag = np.zeros((4, 1, 16))

    for k in range(0, 16):
        t_X = gen_16QAM_symbol_vec(4)
        for i in range(0, 4):
            X_ref_real[i][0][k] = t_X[i][0].real.astype(np.float32)
            X_ref_imag[i][0][k] = t_X[i][0].imag.astype(np.float32)


