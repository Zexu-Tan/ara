#ifndef DATASET_H
#define DATASET_H


// NO trailing semicolons on these!
#define NT          4
#define NR          4
#define N_BEAM      1024
#define N_ITR_BEAM  1024
#define SNR         20

// declaration only — no storage allocated here


extern double H_real[NT][NR][N_BEAM];
extern double H_imag[NT][NR][N_BEAM];

extern double Gram_ref_real[NT][NR][N_BEAM];
extern double Gram_ref_imag[NT][NR][N_BEAM];

extern double L_ref_real[NT][NR][N_BEAM];
extern double L_ref_imag[NT][NR][N_BEAM];

extern double Y_ref_real[NT][1][N_BEAM];
extern double Y_ref_imag[NT][1][N_BEAM];

extern double Z_ref_real[NT][1][N_BEAM];
extern double Z_ref_imag[NT][1][N_BEAM];

extern double U_ref_real[NT][1][N_BEAM];
extern double U_ref_imag[NT][1][N_BEAM];

extern double X_ref_real[NT][1][N_BEAM];
extern double X_ref_imag[NT][1][N_BEAM];


#endif  // DATASET_H
