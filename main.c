#include "compute_check_to_value_msg.h"
#include <math.h>
#include <float.h>

//size neigh > 0 y size_syn > 0
//esta función dada una matriz L de mensajes calcula los mensajes que reciben los value nodes de los check nodes y por tanto equivalen al mensaje del "check node"
void compute_check_to_value(float L[CHECK][VNODES], int* syndrome, float out[CHECK][VNODES], int size_checks,int size_vnode){
    

    for(int i = 0; i < CHECK; i++){
        if(i == size_checks) break;
   
        float min1 = FLT_MAX,min2 = FLT_MAX;
        int minpos = -1;
        int row_sign = 1;
        float product = 1.0;
        //buscamos los minimos 
        for(int j = 0; j < VNODES; j++){
            float val_i_j = L[i][j];
            float absolute_value = fabs(val_i_j);
            if(j == size_vnode) break;
            if(absolute_value < min1 && val_i_j  != 0){
                min2 = min1;
                min1 = absolute_value;
                minpos = j;
            }else if (absolute_value < min2 && val_i_j != 0) {
                min2 = absolute_value;
            }
            row_sign = row_sign ^ signbit(val_i_j);
            //(product = product * (signbit(L[i][j]) ^ syndrome[i]); 
        }
        for(int j = 0; j < VNODES; j++){
            if(j == size_vnode) break;
            float val_i_j = L[i][j];
            int sign_j = row_sign^signbit(val_i_j) ^ syndrome[i];
            float final_sign = (sign_j == 0) ? +1.0f : -1.0f;
            float magnitude;
            if(j != minpos){
                //calcular el signo de la fila y luego para cada columna multiplicar su signo (en 0 y 1) por el signo de la fila en tera es lo mismo que calcular el signo
                // de toda la fila sin ese elemento de la columna j
                //L[i][j] = min1 * final_sign;
                magnitude = min1;
            }
            else{
                //L[i][j] = min2 * final_sign;
                magnitude = min2;
            }

            if (val_i_j != 0.0f) {
                out[i][j] = final_sign * magnitude;
            } else {
                out[i][j] = 0.0f;
            }
        }
        
    }
    
}