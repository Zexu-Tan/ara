#ifndef DATASET_H
#define DATASET_H

// NO trailing semicolons on these!
#define Nt        4
#define Nr        4
#define N_BEAM    16
#define N_ITR_BEAM 16

// declaration only — no storage allocated here
extern float H_real[Nt][Nr][N_BEAM];
extern float H_imag[Nt][Nr][N_BEAM];

#endif  // DATASET_H
