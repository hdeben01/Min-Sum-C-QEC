#include "compute_check_to_value_msg.h"
#include <math.h>
#include <float.h>
#include <stdio.h>

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

int main() {
    float L[CHECK][VNODES];
    FILE *file = fopen("input.txt","r");
    if (file == NULL){
        perror("Error abriendo el archivo");
        return 1;
    }

    //leemos las columnas y filas
    int rows,cols;
    if (fscanf(file, "%d %d", &rows, &cols) != 2) {
        fprintf(stderr, "Error leyendo dimensiones\n");
        fclose(file);
        return 1;
    }

    //leemos la matriz L (beliefs iniciales)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (fscanf(file, "%f", &L[i][j]) != 1) {
                fprintf(stderr, "Error leyendo valor en fila %d columna %d\n", i, j);
                fclose(file);
                return 1;
            }
        }
    }

    int syndrome[CHECK];
     // Leer el síndrome
    for (int j = 0; j < cols; j++) {
        if (fscanf(file, "%d", &syndrome[j]) != 1) {
            fprintf(stderr, "Error leyendo syndrome[%d]\n", j);
            fclose(file);
            return 1;
        }
    }
    float out[CHECK][VNODES];

    compute_check_to_value(L, syndrome, out, rows, cols);

    // Mostrar resultados
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.6f ", out[i][j]);
        }
        printf("\n");
    }

    return 0;
}