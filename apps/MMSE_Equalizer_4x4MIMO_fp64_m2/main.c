// Author: Zexu.Tan <zexu.tan@tu-dresden.de>

#include <stdint.h>
#include <string.h>
#include <stddef.h>
#include <riscv_vector.h>

#include "runtime.h"
#include <math.h>

#ifndef SPIKE
#include "printf.h"
#else
#include <stdio.h>
#endif

#include "kernel/arith_lib.h"
#include "kernel/dataset.h"

//#define EPS 1e-1f //0.1

//#define DEBUG 1

int main(void) {
    size_t vl = vsetvl_e64m2(N_ITR_BEAM);
    double Gram_real[NT][NR][N_BEAM] = {0};
    double Gram_imag[NT][NR][N_BEAM] = {0};

    double sigma_n2 = 1.0f / powf(10.0f, SNR/10.0f);

    start_timer();
    v_compute_gram_fp32(NT, NR, vl, sigma_n2, (double*)H_real, (double*)H_imag, (double*)Gram_real, (double*)Gram_imag);
    stop_timer();

    int64_t runtime = get_timer();

    float performance = (8.0 * NT * (NT + 1) / 2 - 4.0 * NT)* NT * vl / runtime;

    float utilization = 100.0 * performance / (2.0 * NR_LANES);

     
    printf("The execution took %d cycles.\n", runtime);
    printf("The performance is %f DP-FLOP/cycle (%f%% utilization).\n", performance, utilization);
    printf("========Statistics========\n");
    printf("vl = %d\n", vl);
    printf("N_BEAM = %d\n", N_BEAM);
    printf("NR_LANES = %d\n", NR_LANES);
    printf("SNR_db = %d\n", SNR);
    printf("The execution took %d cycles.\n", runtime);
    printf("==========================\n");

    


#ifdef DEBUG

    int k_debug = 0;
    
    printf("========Kernel: MMSE_Equalizer_4x4MIMO_FP32========\n");

    printf("Sigma_n2 = %f \n", sigma_n2);

    
    printf("=======real\n");
    for (int i = 0; i < NR; i++){
        for (int j = 0; j < NT; j++){
            printf("%f ", Gram_real[i][j][k_debug]);
        }
        printf("\n");
    }

    printf("=======real_ref\n");

    for (int i = 0; i < NR; i++){
        for (int j = 0; j < NT; j++){
            printf("%f ", Gram_ref_real[i][j][k_debug]);
        }
        printf("\n");
    }
    

    printf("=======imag\n");
    for (int i = 0; i < NR; i++){
        for (int j = 0; j < NT; j++){
            printf("%f ", Gram_imag[i][j][k_debug]);
        }
        printf("\n");
    }

    printf("=======imag_ref\n");

    for (int i = 0; i < NR; i++){
        for (int j = 0; j < NT; j++){
            printf("%f ", ((double*)Gram_ref_imag)[GET_IDX(i, j, k_debug, NR, vl)]);
        }
        printf("\n");
    }

    
/*
    printf("===================LL_Decomp=====================\n");

    printf("=======real\n");
    printf("\n");
    for (int i = 0; i < NR; i++){
        for (int j = 0; j < NT; j++){
            printf("%f ", L_real[i][j][k_debug]);
        }
        printf("\n");
    }

    printf("=======real_ref\n");
    printf("\n");
    for (int i = 0; i < NR; i++){
        for (int j = 0; j < NT; j++){
            printf("%f ", L_ref_real[i][j][k_debug]);
        }
        printf("\n");
    }


    printf("=======compare real_ref\n");
    printf("\n");
    for (int i = 0; i < NR; i++){
        for (int j = 0; j < NT; j++){
            if (fabs(L_ref_real[i][j][k_debug] -  L_real[i][j][k_debug]) > 0.1) {
                printf("ERROR  ");
            }
            else {
                printf("TRUE  ");
            }
            //printf("%f ", L_ref_real[i][j][k_debug]);
        }
        printf("\n");
    }

    printf("=======imag\n");
    printf("\n");
    for (int i = 0; i < NR; i++){
        for (int j = 0; j < NT; j++){
            printf("%f ", L_imag[i][j][k_debug]);
        }
        printf("\n");
    }

    printf("=======imag_ref\n");
    printf("\n");
    for (int i = 0; i < NR; i++){
        for (int j = 0; j < NT; j++){
            printf("%f ", L_imag[i][j][k_debug]);
        }
        printf("\n");
    }


    printf("===================Compute Z=====================\n");
    
    
    printf("=======Z_real\n");
    for (int i = 0; i < NT; i++){
        for (int j = 0; j < 1; j++){
            printf("%f ", z_real[i][j][k_debug]);
        }
        printf("\n");
    }

    printf("=======Z_real_ref\n");

    for (int i = 0; i < NT; i++){
        for (int j = 0; j < 1; j++){
            printf("%f ", Z_ref_real[i][j][k_debug]);
        }
        printf("\n");
    }

    printf("=======Z_imag\n");
    for (int i = 0; i < NT; i++){
        for (int j = 0; j < 1; j++){
            printf("%f ", z_imag[i][j][k_debug]);
        }
        printf("\n");
    }

    printf("=======Z_imag_ref\n");

    for (int i = 0; i < NT; i++){
        for (int j = 0; j < 1; j++){
            printf("%f ", Z_ref_imag[i][j][k_debug]);
        }
        printf("\n");
    }

    printf("===================Forward Substitution=====================\n");
    
    
    printf("=======u_real\n");
    for (int i = 0; i < NT; i++){
        for (int j = 0; j < 1; j++){
            printf("%f ", u_real[i][j][k_debug]);
        }
        printf("\n");
    }

    
    printf("=======U_real_ref\n");

    for (int i = 0; i < NT; i++){
        for (int j = 0; j < 1; j++){
            printf("%f ", U_ref_real[i][j][k_debug]);
        }
        printf("\n");
    }
    

    printf("=======u_imag\n");
    for (int i = 0; i < NT; i++){
        for (int j = 0; j < 1; j++){
            printf("%f ", u_imag[i][j][k_debug]);
        }
        printf("\n");
    }


    
    printf("=======U_imag_ref\n");

    for (int i = 0; i < NT; i++){
        for (int j = 0; j < 1; j++){
            printf("%f ", U_ref_imag[i][j][k_debug]);
        }
        printf("\n");
    }
    
    
    printf("===================Backward Substitution=====================\n");
    
    
    printf("=======x_real\n");
    for (int i = 0; i < NT; i++){
        for (int j = 0; j < 1; j++){
            printf("%f ", x_real[i][j][k_debug]);
        }
        printf("\n");
    }

    
    printf("=======X_real_ref\n");

    for (int i = 0; i < NT; i++){
        for (int j = 0; j < 1; j++){
            printf("%f ", X_ref_real[i][j][k_debug]);
        }
        printf("\n");
    }
    

    printf("=======x_imag\n");
    for (int i = 0; i < NT; i++){
        for (int j = 0; j < 1; j++){
            printf("%f ", x_imag[i][j][k_debug]);
        }
        printf("\n");
    }


    
    printf("=======X_imag_ref\n");

    for (int i = 0; i < NT; i++){
        for (int j = 0; j < 1; j++){
            printf("%f ", X_ref_imag[i][j][k_debug]);
        }
        printf("\n");
    }

    
    
*/
    
    
#endif    

  
  
return 0;
}
