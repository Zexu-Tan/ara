#ifndef DATASET_H
#define DATASET_H

// NO trailing semicolons on these!
#define NT        4
#define NR        4
#define N_BEAM    16
#define N_ITR_BEAM 16

// declaration only — no storage allocated here
extern float H_real[NT][NR][N_BEAM];
extern float H_imag[NT][NR][N_BEAM];

#endif  // DATASET_H
