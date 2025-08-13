#!/usr/bin/env python3

import numpy as np
import argparse
import sys
import os


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
    print(f'float {var_name}[NT][NR][N_BEAM] = {{')
    for t in range(NT):
        print('  {')  # start Nt block
        for r in range(NR):
            row = ", ".join(f"{val:.8f}f" for val in H[t, r])
            print(f'    {{ {row} }},')
        print('  },')
    print('};')

def emit_c_definition(H, header, var_name="H"):
    """
    Emit the complete C source: include + array.
    """
    emit_include(header)
    emit_array_definition(H, var_name = "H_real")
    emit_array_definition(H, var_name = "H_imag")


def main():
    p = argparse.ArgumentParser(description="Generate C file defining float H[NT][NR][N_BEAM].")
    p.add_argument("--NT",     type=int, default=4,   help="First dimension (default 4)")
    p.add_argument("--NR",     type=int, default=4,   help="Second dimension (default 4)")
    p.add_argument("--N_BEAM", type=int, default=16,  help="Third dimension (default 64)")
    p.add_argument("--seed",   type=int, default=None,help="RNG seed for reproducibility")
    p.add_argument("--out",    type=str, default=None,help="Output C file (stdout if omitted)")
    p.add_argument("--header", type=str, default="dataset.h", help="Header to include (default dataset.h)")
    args = p.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    # Generate random floats in [0,1)
    H_real = (np.random.rand(args.NT, args.NR, args.N_BEAM) / np.sqrt(2)).astype(np.float32)
    H_imag = (np.random.rand(args.NT, args.NR, args.N_BEAM) / np.sqrt(2)).astype(np.float32)

    # Redirect stdout if needed
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            orig = sys.stdout
            sys.stdout = f
            #emit_c_definition(H, args.header)

            emit_include(args.header, args.NT, args.NR, args.N_BEAM)
            emit_array_definition(H_real, var_name = "H_real")
            emit_array_definition(H_imag, var_name = "H_imag")

            sys.stdout = orig
    else:
        emit_include(args.header, args.Nt, args.Nr, args.N_BEAM)
        emit_array_definition(H_real, var_name="H_real")
        emit_array_definition(H_imag, var_name="H_imag")

if __name__ == "__main__":
    main()
