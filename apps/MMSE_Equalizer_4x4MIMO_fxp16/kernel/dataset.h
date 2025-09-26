#include <stdint.h>

#ifndef DATASET_H
#define DATASET_H

#define NT 4
#define NR 4
#define SNR 10
#define N_SUBCARRIER 128
#define N_SUBCARRIER_ITR 128

extern int16_t sigma_n2_fxp;

extern int16_t H_real_fxp [NT][NR][N_SUBCARRIER];
extern int16_t H_imag_fxp [NT][NR][N_SUBCARRIER];

extern int16_t Gram_ref_real [NT][NR][N_SUBCARRIER];
extern int16_t Gram_ref_imag [NT][NR][N_SUBCARRIER];

extern int16_t L_ref_real [NT][NR][N_SUBCARRIER];
extern int16_t L_ref_imag [NT][NR][N_SUBCARRIER];

extern int16_t Y_ref_real [NT][1][N_SUBCARRIER];
extern int16_t Y_ref_imag [NT][1][N_SUBCARRIER];

extern int16_t Z_ref_real [NT][1][N_SUBCARRIER];
extern int16_t Z_ref_imag [NT][1][N_SUBCARRIER];

extern int16_t U_ref_real [NT][1][N_SUBCARRIER];
extern int16_t U_ref_imag [NT][1][N_SUBCARRIER];

extern int16_t X_ref_real [NT][1][N_SUBCARRIER];
extern int16_t X_ref_imag [NT][1][N_SUBCARRIER];

extern int16_t X_est_ref_real [NT][1][N_SUBCARRIER];
extern int16_t X_est_ref_imag [NT][1][N_SUBCARRIER];


#endif
