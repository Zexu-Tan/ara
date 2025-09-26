#!/usr/bin/env python3
import numpy as np
import argparse
import sys
import os

from helper.fxp_lib import *


def emit_var_definition(var, var_name="var", var_type="int16_t"):
    """
    Emit the C definition of the 3D float array H.
    """
    print(f'{var_type} {var_name} = {var} ;')


def emit_array_definition(H, var_name="H", var_type="int16_t"):
    """
    Emit the C definition of the 3D float array H.
    """
    NT, NR, N_SUBCARRIER = H.shape
    print(f'{var_type} {var_name}[{NT}][{NR}][{N_SUBCARRIER}] = {{')
    for t in range(NT):
        print('  {')  # start Nt block
        for r in range(NR):
            row = ", ".join(f"{val}" for val in H[t, r])
            print(f'    {{ {row} }},')
        print('  },')
    print('};\n')


def main():
    p = argparse.ArgumentParser(description="Generate Header and C file for Kernel MMSE_Equalizer")
    p.add_argument("--NT", type=int, default=4, help="First dimension (default 4)")
    p.add_argument("--NR", type=int, default=4, help="Second dimension (default 4)")
    p.add_argument("--WL", type=int, default=16, help="Word Length (default 16)")
    p.add_argument("--N_SUBCARRIER", type=int, default=128, help="Third dimension (default 128)")
    p.add_argument("--N_SUBCARRIER_ITR", type=int, default=128, help="Third dimension (default 128)")
    p.add_argument("--SNR", type=int, default=10, help="Signal to Noise Ratio (default 15dB)")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    p.add_argument("--header_out", type=str, default="dataset.h", help="Header to include (default dataset.h)")
    p.add_argument("--c_out", type=str, default="dataset.c", help="Output C file (stdout if omitted)")
    args = p.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    assert args.WL in (16, 32, 64)

    M = 0
    N = 0

    if args.NT == 4:
        if args.WL == 16:
            # 4x4MIMO, WL==16, Q5.10
            M = 5
            N = 10
            wl = "int16_t"




    SNR_dB = args.SNR
    linear_SNR = 10 ** (SNR_dB / 10)
    sigma_n2 = 1 / linear_SNR
    sigma_n2_fxp = to_int(fp_to_fxp(sigma_n2, M, N), args.WL)


    X_ref_real = np.zeros((args.NT, 1, args.N_SUBCARRIER))
    X_ref_imag = np.zeros((args.NT, 1, args.N_SUBCARRIER))

    X_ref_real_fxp = np.zeros((args.NT, 1, args.N_SUBCARRIER))
    X_ref_imag_fxp = np.zeros((args.NT, 1, args.N_SUBCARRIER))

    for k in range(0, args.N_SUBCARRIER):
        t_X = gen_16QAM_symbol_vec(args.NT)
        for i in range(0, args.NT):
            X_ref_real[i][0][k] = t_X[i][0].real.astype(np.float64)
            X_ref_imag[i][0][k] = t_X[i][0].imag.astype(np.float64)

            X_ref_real_fxp[i][0][k] = to_int(fp_to_fxp(t_X[i][0].real.astype(np.float64),M, N), args.WL)
            X_ref_imag_fxp[i][0][k] = to_int(fp_to_fxp(t_X[i][0].imag.astype(np.float64),M, N), args.WL)


    H_real = (np.random.rand(args.NT, args.NR, args.N_SUBCARRIER) / np.sqrt(2))
    H_imag = (np.random.rand(args.NT, args.NR, args.N_SUBCARRIER) / np.sqrt(2))

    H_real_fxp = fp_to_fxp(H_real, M, N)
    H_imag_fxp = fp_to_fxp(H_imag, M, N)

    for k in range(0, args.N_SUBCARRIER):
        for i in range(0, args.NT):
            for j in range(0, args.NR):
                H_real_fxp[i][j][k] = to_int(H_real_fxp[i][j][k], args.WL)
                H_imag_fxp[i][j][k] = to_int(H_imag_fxp[i][j][k], args.WL)

    t_H_real = np.zeros((args.NT, args.NR))
    t_H_imag = np.zeros((args.NT, args.NR))
        
    t_H_real_fxp = np.zeros((args.NT, args.NR))
    t_H_imag_fxp = np.zeros((args.NT, args.NR))

    Gram_ref_real = np.zeros_like(H_real)
    Gram_ref_imag = np.zeros_like(H_imag)

    # compute_gram
    for k in range(0, args.N_SUBCARRIER):
        for i in range(0, args.NT):
            for j in range(0, args.NR):
                t_H_real[i][j] = H_real_fxp[i][j][k]
                t_H_imag[i][j] = H_imag_fxp[i][j][k]
        t_H = t_H_real + 1j * t_H_imag
        t_Gram = (t_H.conj().T) @ t_H

        for i in range(0, args.NT):
            for j in range(0, args.NR):
                Gram_ref_real[i][j][k] = to_int((int(np.real(t_Gram[i][j])) >> N), args.WL)
                Gram_ref_imag[i][j][k] = to_int((int(np.imag(t_Gram[i][j])) >> N), args.WL)

        for i in range(0, args.NT):
            for j in range(0, args.NR):
                if i == j:
                    Gram_ref_real[i][j][k] += sigma_n2_fxp

    Gram_ref_real = Gram_ref_real.astype(np.int64)
    Gram_ref_imag = Gram_ref_imag.astype(np.int64)

    # Compute Z
    Y_ref_real = np.zeros((args.NT, 1, args.N_SUBCARRIER))
    Y_ref_imag = np.zeros((args.NT, 1, args.N_SUBCARRIER))
    Y_ref_real_fxp = np.zeros((args.NT, 1, args.N_SUBCARRIER))
    Y_ref_imag_fxp = np.zeros((args.NT, 1, args.N_SUBCARRIER))

    t_Y_fxp_real = np.zeros((args.NT, 1))
    t_Y_fxp_imag = np.zeros((args.NT, 1))

    t_X_real = np.zeros((args.NT, 1))
    t_X_imag = np.zeros((args.NT, 1))

    t_X_real_fxp = np.zeros((args.NT, 1))
    t_X_imag_fxp = np.zeros((args.NT, 1))

    for k in range(0, args.N_SUBCARRIER):
        for i in range(0, args.NT):
            for j in range(0, args.NR):
                t_H_real[i][j] = H_real[i][j][k]
                t_H_imag[i][j] = H_imag[i][j][k]

                t_H_real_fxp[i][j] = H_real_fxp[i][j][k]
                t_H_imag_fxp[i][j] = H_imag_fxp[i][j][k]

        t_H = t_H_real + 1j * t_H_imag
        t_H_fxp = t_H_real_fxp + 1j *t_H_imag_fxp

        for i in range(0, args.NT):
            t_X_real[i][0] = X_ref_real[i][0][k]
            t_X_imag[i][0] = X_ref_imag[i][0][k]

            t_X_real_fxp[i][0] = X_ref_real_fxp[i][0][k]
            t_X_imag_fxp[i][0] = X_ref_imag_fxp[i][0][k]

        t_X = t_X_real + 1j * t_X_imag
        t_X_fxp = t_X_real_fxp + 1j * t_X_imag_fxp

        n = np.sqrt(sigma_n2 / 2) * (np.random.randn(args.NT, 1) + 1j * np.random.randn(args.NT, 1))

        n_fxp = fp_to_fxp((np.real(n)), M, N) + 1j* fp_to_fxp((np.imag(n)), M, N)

        t_Y = t_H @ t_X + n

        t_Y_fxp = t_H_fxp @ t_X_fxp

        for i in range(0, args.NT):
            t_Y_fxp_real[i][0] = to_int(int(np.real(t_Y_fxp[i][0]))>>N, args.WL)
            t_Y_fxp_imag[i][0] = to_int(int(np.imag(t_Y_fxp[i][0]))>>N, args.WL)

            t_Y_fxp_real[i][0]  += n_fxp[i][0].real
            t_Y_fxp_imag[i][0]  += n_fxp[i][0].imag

        for i in range(0, args.NT):
            for j in range(0, 1):
                Y_ref_real[i][j][k] = t_Y[i][j].real.astype(np.float64)
                Y_ref_imag[i][j][k] = t_Y[i][j].imag.astype(np.float64)
                Y_ref_real_fxp[i][j][k] = to_int(t_Y_fxp_real[i][j] , args.WL)
                Y_ref_imag_fxp[i][j][k] = to_int(t_Y_fxp_imag[i][j] , args.WL)

    Z_ref_real_fxp = np.zeros((args.NT, 1, args.N_SUBCARRIER))
    Z_ref_imag_fxp = np.zeros((args.NT, 1, args.N_SUBCARRIER))
    Z_ref_real = np.zeros((args.NT, 1, args.N_SUBCARRIER))
    Z_ref_imag = np.zeros((args.NT, 1, args.N_SUBCARRIER))

    t_Y_real = np.zeros((args.NT, 1))
    t_Y_imag = np.zeros((args.NT, 1))

    t_Z_real = np.zeros((args.NT, 1))
    t_Z_imag = np.zeros((args.NT, 1))

    for k in range(0, args.N_SUBCARRIER):
        for i in range(0, args.NT):
            for j in range(0, args.NR):

                t_H_real[i][j] = H_real[i][j][k]
                t_H_imag[i][j] = H_imag[i][j][k]

                t_H_real_fxp[i][j] = H_real_fxp[i][j][k]
                t_H_imag_fxp[i][j] = H_imag_fxp[i][j][k]

            t_H = t_H_real + 1j * t_H_imag    
            t_H_fxp = t_H_real_fxp + 1j * t_H_imag_fxp

            for i in range(0, args.NT):
                t_Y_real[i][0] = Y_ref_real[i][0][k]
                t_Y_imag[i][0] = Y_ref_imag[i][0][k]

                t_Y_fxp_real[i][0] = Y_ref_real_fxp[i][0][k]
                t_Y_fxp_imag[i][0] = Y_ref_imag_fxp[i][0][k]

            t_Y = t_Y_real + 1j * t_Y_imag    
            t_Y_fxp = t_Y_fxp_real + 1j * t_Y_fxp_imag

        t_Z = (t_H.conj().T) @ t_Y
        t_Z_fxp = (t_H_fxp.conj().T) @ t_Y_fxp

        for i in range(0, args.NT):
            t_Z_real[i][0] = to_int(int(np.real(t_Z_fxp[i][0])) >> N, args.WL)
            t_Z_imag[i][0] = to_int(int(np.imag(t_Z_fxp[i][0])) >> N, args.WL)


        for i in range(0, args.NT):
            for j in range(0, 1):
                Z_ref_real[i][j][k] = t_Z[i][j].real.astype(np.float64)
                Z_ref_imag[i][j][k] = t_Z[i][j].imag.astype(np.float64)
                Z_ref_real_fxp[i][j][k] = to_int(t_Z_real[i][j], args.WL)
                Z_ref_imag_fxp[i][j][k] = to_int(t_Z_imag[i][j], args.WL)





    # LL_decomposition

    Gram_ref = Gram_ref_real + 1j * Gram_ref_imag

    L_ref = np.zeros_like(Gram_ref)


    for k in range(args.N_SUBCARRIER):
        for j in range(args.NR):
            t_Gii_rsqrt_fxp = 0
            for i in range(j, args.NT):
                t_Gij = Gram_ref[i][j][k]
                t_Lij = 0
                for m in range(0, j):
                    t_Lij -= L_ref[i][m][k] * np.conj(L_ref[j][m][k])
                t_Lij = to_int((int(np.real(t_Lij)) >> N), args.WL) + 1j * to_int((int(np.imag(t_Lij)) >> N), args.WL)
                t_Lij = t_Gij + t_Lij

                if i == j:
                    t_Gii_rsqrt_fp = 1 / np.sqrt(fxp_to_fp(np.real(t_Lij), N))

                    t_Gii_rsqrt_fxp = fp_to_fxp(t_Gii_rsqrt_fp, M, N)

                    t_Lij = t_Gii_rsqrt_fxp

                else:
                    t_Lij = t_Lij * t_Gii_rsqrt_fxp

                    t_Lij = to_int((int(np.real(t_Lij)) >> N), args.WL) + 1j* to_int((int(np.imag(t_Lij)) >> N), args.WL)

                L_ref[i][j][k] = t_Lij

    L_ref_real_fxp = np.real(L_ref).astype(np.int64)
    L_ref_imag_fxp = np.imag(L_ref).astype(np.int64)

    L_ref_fxp = L_ref_real_fxp + 1j * L_ref_imag_fxp



    U_ref_fxp = np.zeros((args.NT, 1, args.N_SUBCARRIER), dtype=np.complex64)

    Z_ref_fxp = Z_ref_real_fxp + 1j * Z_ref_imag_fxp


    for k in range(args.N_SUBCARRIER):
        for i in range(args.NT):
            t_u_fxp = Z_ref_fxp[i][0][k]
            t_sum_fxp = 0
            for j in range(0, i):
                t_sum_fxp -= L_ref_fxp[i][j][k] * U_ref_fxp[j][0][k]
            t_sum_fxp = to_int((int(np.real(t_sum_fxp)) >> N), args.WL) + 1j * to_int((int(np.imag(t_sum_fxp)) >> N), args.WL)
            t_u_fxp += t_sum_fxp
            t_u_fxp *= L_ref_fxp[i][i][k]
            t_u_fxp = to_int((int(np.real(t_u_fxp)) >> N), args.WL) + 1j * to_int((int(np.imag(t_u_fxp)) >> N), args.WL)
            U_ref_fxp[i][0][k] = t_u_fxp

    U_ref_real_fxp = np.real(U_ref_fxp).astype(np.int64)
    U_ref_imag_fxp = np.imag(U_ref_fxp).astype(np.int64)

    X_est_fxp = np.zeros_like(U_ref_fxp)
    for k in range(args.N_SUBCARRIER):
        for i in range(args.NT - 1, -1, -1):
            t_x_fxp = U_ref_fxp[i][0][k]
            t_sum_fxp = 0
            for j in range(i+1, args.NT):
                t_sum_fxp -= np.conj(L_ref_fxp[j][i][k]) * X_est_fxp[j][0][k]
            t_sum_fxp = to_int((int(np.real(t_sum_fxp)) >> N), args.WL) + 1j * to_int((int(np.imag(t_sum_fxp)) >> N), args.WL)
            t_x_fxp += t_sum_fxp
            t_x_fxp *= L_ref_fxp[i][i][k]
            t_x_fxp = to_int((int(np.real(t_x_fxp)) >> N), args.WL) + 1j * to_int((int(np.imag(t_x_fxp)) >> N), args.WL)
            X_est_fxp[i][0][k] = t_x_fxp

    X_est_ref_real_fxp = X_est_fxp.real.astype(np.int64)
    X_est_ref_imag_fxp = X_est_fxp.imag.astype(np.int64)




    if args.header_out:
        os.makedirs(os.path.dirname(f"../{args.header_out}") or ".", exist_ok=True)
        with open(f"../{args.header_out}", "w") as f:
            orig = sys.stdout
            sys.stdout = f

            print(f'#include <stdint.h>\n')
            print(f'#ifndef DATASET_H')
            print(f'#define DATASET_H\n')

            print(f"#define NT {args.NT}")
            print(f"#define NR {args.NR}")
            print(f"#define SNR {args.SNR}")
            print(f"#define N_SUBCARRIER {args.N_SUBCARRIER}")
            print(f"#define N_SUBCARRIER_ITR {args.N_SUBCARRIER_ITR}\n")


            print(f"extern {wl} sigma_n2_fxp;\n")

            print(f"extern {wl} H_real_fxp [NT][NR][N_SUBCARRIER];")
            print(f"extern {wl} H_imag_fxp [NT][NR][N_SUBCARRIER];\n")

            print(f"extern {wl} Gram_ref_real [NT][NR][N_SUBCARRIER];")
            print(f"extern {wl} Gram_ref_imag [NT][NR][N_SUBCARRIER];\n")

            print(f"extern {wl} L_ref_real [NT][NR][N_SUBCARRIER];")
            print(f"extern {wl} L_ref_imag [NT][NR][N_SUBCARRIER];\n")

            print(f"extern {wl} Y_ref_real [NT][1][N_SUBCARRIER];")
            print(f"extern {wl} Y_ref_imag [NT][1][N_SUBCARRIER];\n")

            print(f"extern {wl} Z_ref_real [NT][1][N_SUBCARRIER];")
            print(f"extern {wl} Z_ref_imag [NT][1][N_SUBCARRIER];\n")

            print(f"extern {wl} U_ref_real [NT][1][N_SUBCARRIER];")
            print(f"extern {wl} U_ref_imag [NT][1][N_SUBCARRIER];\n")

            print(f"extern {wl} X_ref_real [NT][1][N_SUBCARRIER];")
            print(f"extern {wl} X_ref_imag [NT][1][N_SUBCARRIER];\n")

            print(f"extern {wl} X_est_ref_real [NT][1][N_SUBCARRIER];")
            print(f"extern {wl} X_est_ref_imag [NT][1][N_SUBCARRIER];\n")

            print(f'\n#endif')

            sys.stdout = orig

    if args.c_out:
        os.makedirs(os.path.dirname(f"../{args.c_out}") or ".", exist_ok=True)
        with open(f"../{args.c_out}", "w") as f:
            orig = sys.stdout
            sys.stdout = f
            print(f'#include "{args.header_out}"\n')

            emit_var_definition(sigma_n2_fxp, "sigma_n2_fxp", wl)
            emit_array_definition(H_real_fxp, "H_real_fxp", wl)
            emit_array_definition(H_imag_fxp, "H_imag_fxp", wl)
            emit_array_definition(Gram_ref_real, "Gram_ref_real", wl)
            emit_array_definition(Gram_ref_imag, "Gram_ref_imag", wl)
            emit_array_definition(L_ref_real_fxp, "L_ref_real", wl)
            emit_array_definition(L_ref_imag_fxp, "L_ref_imag", wl)
            emit_array_definition(Z_ref_real_fxp, "Z_ref_real", wl)
            emit_array_definition(Z_ref_imag_fxp, "Z_ref_imag", wl)
            emit_array_definition(U_ref_real_fxp, "U_ref_real", wl)
            emit_array_definition(U_ref_imag_fxp, "U_ref_imag", wl)
            emit_array_definition(Y_ref_real_fxp, "Y_ref_real", wl)
            emit_array_definition(Y_ref_imag_fxp, "Y_ref_imag", wl)
            emit_array_definition(X_est_ref_real_fxp, "X_est_ref_real", wl)
            emit_array_definition(X_est_ref_imag_fxp, "X_est_ref_imag", wl)
            emit_array_definition(X_ref_real_fxp, "X_ref_real", wl)
            emit_array_definition(X_ref_imag_fxp, "X_ref_imag", wl)

            sys.stdout = orig


if __name__ == "__main__":
    main()