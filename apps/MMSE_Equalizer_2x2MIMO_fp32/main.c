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

#define DEBUG 1

int main(void) {
    size_t vl = vsetvl_e32m1(N_ITR_BEAM);
    float Gram_real[NT][NR][N_BEAM] = {0};
    float Gram_imag[NT][NR][N_BEAM] = {0};
    
    float L_real[NT][NR][N_BEAM] = {0};
    float L_imag[NT][NR][N_BEAM] = {0};
    float z_real[NT][1][N_BEAM] = {0};
    float z_imag[NT][1][N_BEAM] = {0};
    float u_real[NT][1][N_BEAM] = {0};
    float u_imag[NT][1][N_BEAM] = {0};
    float x_real[NT][1][N_BEAM] = {0};
    float x_imag[NT][1][N_BEAM] = {0};
    float sigma_n2 = 1.0f / powf(10.0f, SNR/10.0f);

    //start_timer();
    v_compute_gram_fp32(NT, NR, vl, sigma_n2, (float*)H_real, (float*)H_imag, (float*)Gram_real, (float*)Gram_imag);
    //stop_timer();

    //int64_t runtime_compute_gram_fp32 = get_timer();

    //start_timer();
    v_LL_decomp_fp32(NT, NR, vl, (float*)Gram_real, (float*)Gram_imag, (float*)L_real, (float*)L_imag);
    //stop_timer();

    //int64_t LL_decomp_f32 = get_timer();

    //start_timer();
    v_compute_z_fp32(NT, NR, vl, (float*)H_real, (float*)H_imag, (float*)Y_ref_real, (float*)Y_ref_imag, (float*)z_real, (float*)z_imag);
    //stop_timer();
    
    //start_timer();
    v_forward_substitution(NT, NR, vl, (float*)z_real, (float*)z_imag, (float*)L_real, (float*)L_imag, (float*) u_real, (float*) u_imag);
    //stop_timer();
    
    //start_timer();
    v_backward_substitution(NT, NR, vl, (float*)u_real, (float*)u_imag, (float*)L_real, (float*)L_imag, (float*)x_real, (float*)x_imag);
    //stop_timer();


    //float performance = (8.0 * NT * (NT + 1) / 2 - 4.0 * NT)* NT * vl / runtime_compute_gram_fp32;

    //float utilization = 100.0 * performance / (4.0 * NR_LANES);

    /*
    printf("The execution took %d cycles.\n", runtime);
    printf("The performance is %f DP-FLOP/cycle (%f%% utilization).\n", performance/2.0, utilization);
    printf("========Statistics========\n");
    printf("vl = %d\n", vl);
    printf("N_BEAM = %d\n", N_BEAM);
    printf("NR_LANES = %d\n", NR_LANES);
    printf("SNR_db = %d\n", SNR);
    printf("The execution took %d cycles.\n", runtime);
    printf("==========================\n");
    */ 
    


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
            printf("%f ", ((float*)Gram_ref_imag)[GET_IDX(i, j, k_debug, NR, vl)]);
        }
        printf("\n");
    }

    

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
            printf("%f ", L_ref_imag[i][j][k_debug]);
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
    
    
    

    
    
#endif    

  
  
return 0;
}
