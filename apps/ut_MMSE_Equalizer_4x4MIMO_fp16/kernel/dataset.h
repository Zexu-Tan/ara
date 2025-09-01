#ifndef DATASET_H
#define DATASET_H

// NO trailing semicolons on these!
#define NT          4
#define NR          4
#define N_BEAM      64
#define N_ITR_BEAM  64
#define SNR         15

// declaration only — no storage allocated here


extern float H_real[NT][NR][N_BEAM];
extern float H_imag[NT][NR][N_BEAM];

extern float Gram_ref_real[NT][NR][N_BEAM];
extern float Gram_ref_imag[NT][NR][N_BEAM];

extern float L_ref_real[NT][NR][N_BEAM];
extern float L_ref_imag[NT][NR][N_BEAM];

extern float Y_ref_real[NT][1][N_BEAM];
extern float Y_ref_imag[NT][1][N_BEAM];

extern float Z_ref_real[NT][1][N_BEAM];
extern float Z_ref_imag[NT][1][N_BEAM];

extern float U_ref_real[NT][1][N_BEAM];
extern float U_ref_imag[NT][1][N_BEAM];

extern float X_ref_real[NT][1][N_BEAM];
extern float X_ref_imag[NT][1][N_BEAM];





#endif  // DATASET_H
