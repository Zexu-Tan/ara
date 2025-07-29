#!/usr/bin/env python3
"""
generate_H.py

Generate a 3D float array H[Nt][Nr][N_BEAM] with random values
and emit a C source file defining it:

    #include "dataset.h"
    float H[Nt][Nr][N_BEAM] = {
      { { ... }, ... },  // Nt blocks
      ...
    };

Usage:
    ./generate_H.py \
        [--Nt 4] [--Nr 4] [--N_BEAM 64] \
        [--seed 1234] \
        [--out dataset.c]
"""
import numpy as np
import argparse
import sys
import os


def emit_c_definition(H, header, var_name="H"):
    Nt, Nr, N_BEAM = H.shape
    print(f'#include "{header}"\n')
    print(f'// Auto-generated definition of H[{""}{Nt}][{Nr}][{N_BEAM}]')
    print(f'float {var_name}[Nt][Nr][N_BEAM] = {{')
    for t in range(Nt):
        print("  {")  # start Nt block
        for r in range(Nr):
            row = ", ".join(f"{val:.8f}f" for val in H[t, r])
            print(f"    {{ {row} }},")
        print("  },")
    print("};")

def main():
    p = argparse.ArgumentParser(description="Generate C file defining float H[Nt][Nr][N_BEAM].")
    p.add_argument("--Nt",     type=int, default=4,   help="First dimension (default 4)")
    p.add_argument("--Nr",     type=int, default=4,   help="Second dimension (default 4)")
    p.add_argument("--N_BEAM", type=int, default=16,  help="Third dimension (default 64)")
    p.add_argument("--seed",   type=int, default=None,help="RNG seed for reproducibility")
    p.add_argument("--out",    type=str, default=None,help="Output C file (stdout if omitted)")
    p.add_argument("--header", type=str, default="dataset.h", help="Header to include (default dataset.h)")
    args = p.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    # Generate random floats in [0,1)
    H = np.random.rand(args.Nt, args.Nr, args.N_BEAM).astype(np.float32)

    # Redirect stdout if needed
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            orig = sys.stdout
            sys.stdout = f
            emit_c_definition(H, args.header)
            sys.stdout = orig
    else:
        emit_c_definition(H, args.header)

if __name__ == "__main__":
    main()

