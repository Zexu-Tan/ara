#!/usr/bin/env python3
import numpy as np
import argparse
import sys
import os
from gen_16QAM_symbol_vec import gen_16QAM_symbol_vec


def emit_include(header, Nt=None, Nr=None, N_BEAM=None):
    """
    Emit the include directive for the C file.
    """
    print(f'#include "{header}"\n')
    print(f'// Auto-generated definition of H[{Nt}][{Nr}][{N_BEAM}]')


def emit_array_definition(H, var_name="H"):
    """
    Emit the C definition of the 3D float array H.
    """
    NT, NR, N_BEAM = H.shape
    print(f'float {var_name}[{NT}][{NR}][{N_BEAM}] = {{')
    for t in range(NT):
        print('  {')  # start Nt block
        for r in range(NR):
            row = ", ".join(f"{val:.8f}f" for val in H[t, r])
            print(f'    {{ {row} }},')
        print('  },')
    print('};')


def emit_definition_fp_var(var, var_name="var"):
    """
    Emit the C definition of the 3D float array H.
    """
    print(f'float {var_name} = {var} ;')



def emit_c_definition(H, header, var_name="H"):
    """
    Emit the complete C source: include + array.
    """
    emit_include(header)
    emit_array_definition(H, var_name="H_real")
    emit_array_definition(H, var_name="H_imag")



def main():
    p = argparse.ArgumentParser(description="Generate C file defining float H[NT][NR][N_BEAM].")
    p.add_argument("--NT", type=int, default=8, help="First dimension (default 4)")
    p.add_argument("--NR", type=int, default=8, help="Second dimension (default 4)")
    p.add_argument("--N_BEAM", type=int, default=64, help="Third dimension (default 128)")
    p.add_argument("--SNR",type=int, default = 15, help = "Signal to Noise Ratio (default 15dB)")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    p.add_argument("--out", type=str, default="dataset.c", help="Output C file (stdout if omitted)")
    p.add_argument("--header", type=str, default="dataset.h", help="Header to include (default dataset.h)")
    args = p.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    # Generate random floats in [0, 1 / sqrt(2)]
    H_real = (np.random.rand(args.NT, args.NR, args.N_BEAM) / np.sqrt(2)).astype(np.float32)
    H_imag = (np.random.rand(args.NT, args.NR, args.N_BEAM) / np.sqrt(2)).astype(np.float32)

    t_H_real = np.zeros((args.NT, args.NR))
    t_H_imag = np.zeros((args.NT, args.NR))

    linear_SNR = 10 ** (args.SNR/ 10)
    sigma_n2 = 1 / linear_SNR

    X_ref_real = np.zeros((args.NT, 1, args.N_BEAM))
    X_ref_imag = np.zeros((args.NT, 1, args.N_BEAM))

    for k in range(0, args.N_BEAM):
        t_X = gen_16QAM_symbol_vec(args.NT)
        for i in range(0, args.NT):
            X_ref_real[i][0][k] = t_X[i][0].real.astype(np.float32)
            X_ref_imag[i][0][k] = t_X[i][0].imag.astype(np.float32)

    t1_X_real = np.zeros((args.NT, 1))
    t1_X_imag = np.zeros((args.NT, 1))

    Y_ref_real = np.zeros((args.NT, 1, args.N_BEAM))
    Y_ref_imag = np.zeros((args.NT, 1, args.N_BEAM))

    for k in range(0, args.N_BEAM):
        for i in range(0, args.NT):
            for j in range(0, args.NR):
                t_H_real[i][j] = H_real[i][j][k]
                t_H_imag[i][j] = H_imag[i][j][k]
        t_H = t_H_real + 1j * t_H_imag

        for i in range(0, args.NT):
            t1_X_real[i][0] = X_ref_real[i][0][k]
            t1_X_imag[i][0] = X_ref_imag[i][0][k]
        t1_X = t1_X_real + 1j * t1_X_imag

        n = np.sqrt(sigma_n2 / 2) * (np.random.randn(args.NT, 1) + 1j * np.random.randn(args.NT, 1))

        t_Y = t_H @ t1_X + n

        for i in range(0, args.NT):
            for j in range(0, 1):
                Y_ref_real[i][j][k] = t_Y[i][j].real.astype(np.float32)
                Y_ref_imag[i][j][k] = t_Y[i][j].imag.astype(np.float32)

    Z_ref_real = np.zeros((args.NT, 1, args.N_BEAM))
    Z_ref_imag = np.zeros((args.NT, 1, args.N_BEAM))
    t1_Y_real = np.zeros((args.NT, 1))
    t1_Y_imag = np.zeros((args.NT, 1))

    for k in range(0, args.N_BEAM):
        for i in range(0, args.NT):
            for j in range(0, args.NR):
                t_H_real[i][j] = H_real[i][j][k]
                t_H_imag[i][j] = H_imag[i][j][k]
        t_H = t_H_real + 1j * t_H_imag

        for i in range(0, args.NT):
            t1_Y_real[i][0] = Y_ref_real[i][0][k]
            t1_Y_imag[i][0] = Y_ref_imag[i][0][k]
        t1_Y = t1_Y_real + 1j * t1_Y_imag

        t_Z = (t_H.conj().T) @ t1_Y

        for i in range(0, args.NT):
            for j in range(0, 1):
                Z_ref_real[i][j][k] = t_Z[i][j].real.astype(np.float32)
                Z_ref_imag[i][j][k] = t_Z[i][j].imag.astype(np.float32)

    Gram_ref_real = np.zeros_like(H_real)
    Gram_ref_imag = np.zeros_like(H_imag)

    for k in range(0, args.N_BEAM):
        for i in range(0, args.NT):
            for j in range(0, args.NR):
                t_H_real[i][j] = H_real[i][j][k]
                t_H_imag[i][j] = H_imag[i][j][k]
        t_H = t_H_real + 1j * t_H_imag

        #if k == 0:
        #    print(t_H)
        #    print(t_H.conj().T)
        #    print((t_H.conj().T) @ t_H)
        t_Gram = (t_H.conj().T) @ t_H

        for i in range(0, args.NT):
            for j in range(0, args.NR):
                Gram_ref_real[i][j][k] = t_Gram[i][j].real.astype(np.float32)
                Gram_ref_imag[i][j][k] = t_Gram[i][j].imag.astype(np.float32)

    # + σ²I
    for k in range(0, args.N_BEAM):
        for i in range(0, args.NT):
            for j in range(0, args.NR):
                if i == j:
                    Gram_ref_real[i][j][k] = Gram_ref_real[i][j][k] + sigma_n2

    L_ref = np.zeros((args.NT, args.NR, args.N_BEAM), dtype=np.complex64)

    Gram_ref = Gram_ref_real + 1j * Gram_ref_imag



    for k in range(0, args.N_BEAM):
    #for k in range(0, 1):
        for j in range(0, args.NR):
            t_Lii = 0
            for i in range(j, args.NT):
                #print(i, j)
                t_Aij = Gram_ref[i][j][k]
                p0 = 0
                for t_j in range(0, j):
                    p0 += L_ref[i][t_j][k] * np.conj(L_ref[j][t_j][k])
                    #print(f" {(i, t_j), (j, t_j)}")
                t_Aij -= p0

                if i == j:
                    t_Lii = np.sqrt(t_Aij)

                L_ref[i][j][k] = t_Aij / t_Lii
            #print()

    L_ref_real = L_ref.real.astype(np.float32)
    L_ref_imag = L_ref.imag.astype(np.float32)


    u_ref = np.zeros((args.NT, 1, args.N_BEAM), dtype=np.complex64)

    Z_ref = Z_ref_real + 1j * Z_ref_imag


    for k in range(0, args.N_BEAM):
        for i in range(0, args.NT):
            t_z = Z_ref[i][0][k]
            #print(f"{i}")
            for j in range(0, i):
                #print(f" L({i}, {j}), u[{j}] / L({i}, {i})")
                t_z -= L_ref[i][j][k] * u_ref[j][0][k]
            t_z = t_z / L_ref_real[i][i][k]
            u_ref[i][0][k] = t_z

    u_ref_real = u_ref.real.astype(np.float32)
    u_ref_imag = u_ref.imag.astype(np.float32)

    estimated_x_ref = np.zeros((args.NT, 1, args.N_BEAM), dtype=np.complex64)

    for k in range(0, args.N_BEAM):
        for i in range(args.NT-1, -1, -1):
            t_u =u_ref[i][0][k]
            #print(f"{i} u{i}")
            for j in range(i+1, args.NT):
                #print(f" L({j}, {i}), x[{j}] / L({i}, {i})")
                t_u -= np.conj(L_ref[j][i][k]) * estimated_x_ref[j][0][k]
            t_u = t_u / L_ref_real[i][i][k]
            estimated_x_ref[i][0][k] = t_u

    estimated_x_ref_real = estimated_x_ref.real.astype(np.float32)
    estimated_x_ref_imag = estimated_x_ref.imag.astype(np.float32)



    # Redirect stdout if needed
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            orig = sys.stdout
            sys.stdout = f
            # emit_c_definition(H, args.header)
            emit_include(args.header, args.NT, args.NR, args.N_BEAM)
            emit_definition_fp_var(sigma_n2, var_name="sigma_n2")
            emit_array_definition(Y_ref_real, var_name="Y_ref_real")
            emit_array_definition(Y_ref_imag, var_name="Y_ref_imag")
            emit_array_definition(H_real, var_name="H_real")
            emit_array_definition(H_imag, var_name="H_imag")
            emit_array_definition(Gram_ref_real, var_name="Gram_ref_real")
            emit_array_definition(Gram_ref_imag, var_name="Gram_ref_imag")
            emit_array_definition(L_ref_real, var_name="L_ref_real")
            emit_array_definition(L_ref_imag, var_name="L_ref_imag")
            emit_array_definition(Z_ref_real, var_name="Z_ref_real")
            emit_array_definition(Z_ref_imag, var_name="Z_ref_imag")
            emit_array_definition(u_ref_real, var_name="U_ref_real")
            emit_array_definition(u_ref_imag, var_name="U_ref_imag")
            emit_array_definition(estimated_x_ref_real, var_name="X_ref_real")
            emit_array_definition(estimated_x_ref_imag, var_name="X_ref_imag")
            sys.stdout = orig
    else:
        emit_include(args.header, args.NT, args.NR, args.N_BEAM)
        emit_definition_fp_var(sigma_n2, var_name="sigma_n2")
        emit_array_definition(Y_ref_real, var_name="Y_ref_real")
        emit_array_definition(Y_ref_imag, var_name="Y_ref_imag")
        emit_array_definition(H_real, var_name="H_real")
        emit_array_definition(H_imag, var_name="H_imag")
        emit_array_definition(Gram_ref_real, var_name="Gram_ref_real")
        emit_array_definition(Gram_ref_imag, var_name="Gram_ref_imag")
        emit_array_definition(L_ref_real, var_name="L_ref_real")
        emit_array_definition(L_ref_imag, var_name="L_ref_imag")
        emit_array_definition(Z_ref_real, var_name="Z_ref_real")
        emit_array_definition(Z_ref_imag, var_name="Z_ref_imag")
        emit_array_definition(u_ref_real, var_name="U_ref_real")
        emit_array_definition(u_ref_imag, var_name="U_ref_imag")
        emit_array_definition(estimated_x_ref_real, var_name="X_ref_real")
        emit_array_definition(estimated_x_ref_imag, var_name="X_ref_imag")


if __name__ == "__main__":
    main()


