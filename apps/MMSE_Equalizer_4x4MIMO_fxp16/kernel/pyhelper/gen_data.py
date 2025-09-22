#!/usr/bin/env python3
import numpy as np
import argparse
import sys
import os

from helper.fxp_lib import *



def emit_var_definition(var, var_name="var", var_type = "int16_t"):
    """
    Emit the C definition of the 3D float array H.
    """
    print(f'{var_type} {var_name} = {var} ;')

def emit_array_definition(H, var_name="H", var_type="int16_t"):
    """
    Emit the C definition of the 3D float array H.
    """
    NT, NR, N_BEAM = H.shape
    print(f'{var_type} {var_name}[{NT}][{NR}][{N_BEAM}] = {{')
    for t in range(NT):
        print('  {')  # start Nt block
        for r in range(NR):
            row = ", ".join(f"{val}" for val in H[t, r])
            print(f'    {{ {row} }},')
        print('  },')
    print('};\n')


def main():
    p = argparse.ArgumentParser(description="Generate Header and C file for Kernel MMSE_Equalizer")
    p.add_argument("--NT", type=int, default= 4, help="First dimension (default 4)")
    p.add_argument("--NR", type=int, default= 4, help="Second dimension (default 4)")
    p.add_argument("--WL", type=int, default= 16, help = "Word Length (default 16)")
    p.add_argument("--N_SUBCARRIER", type=int, default= 16, help="Third dimension (default 128)")
    p.add_argument("--N_SUBCARRIER_ITR", type=int, default= 16, help="Third dimension (default 128)")
    p.add_argument("--SNR",type=int, default = 20, help = "Signal to Noise Ratio (default 15dB)")
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
        #4x4MIMO, WL==16, Q5.10
        M = 5
        N = 10
      
        
    SNR_dB = 20
    linear_SNR = 10 ** (SNR_dB / 10)
    sigma_n2 = 1 / linear_SNR
    sigma_n2_fxp = to_int(fp_to_fxp(sigma_n2, M, N), args.WL)

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

    Gram_ref_real = np.zeros_like(H_real)
    Gram_ref_imag = np.zeros_like(H_imag)

    #compute_gram
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

    #
    Gram_ref_real = Gram_ref_real.astype(np.int64)
    Gram_ref_imag = Gram_ref_imag.astype(np.int64)














    


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

        
        wl = "int16_t"
        if args.WL == 16:
          wl = "int16_t"


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
        print(f"extern {wl} U_ref_imag [NT][1][N_SUBCARRIER];\n")

        print(f'\n#endif')


        sys.stdout = orig
    
    
    if args.c_out:
      os.makedirs(os.path.dirname(f"../{args.c_out}") or ".", exist_ok=True)
      with open(f"../{args.c_out}", "w") as f:
        orig = sys.stdout
        sys.stdout = f
        print(f'#include "{args.header_out}"\n')

        emit_var_definition(sigma_n2_fxp, "sigma_n2_fxp", wl)
        emit_array_definition(H_real_fxp, "H_real_fxp",wl)
        emit_array_definition(H_imag_fxp, "H_imag_fxp",wl)
        emit_array_definition(Gram_ref_real, "Gram_ref_real", wl)
        emit_array_definition(Gram_ref_imag, "Gram_ref_imag", wl)
        
        sys.stdout = orig
    

if __name__ == "__main__":
    main()