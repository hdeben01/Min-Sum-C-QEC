#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include "compute_check_to_value_msg.h"



void compute_row_operations(float L[CHECK][VNODES], int* syndrome, int size_checks, int size_vnode);
void compute_col_operations(float L[CHECK][VNODES], int* syndrome, int size_checks, int size_vnode, float alpha, float Lj[VNODES], float sum[VNODES]);

//size neigh > 0 y size_syn > 0
//esta función dada una matriz L de mensajes calcula los mensajes que reciben los value nodes de los check nodes y por tanto equivalen al mensaje del "check node"
void compute_check_to_value(float L[CHECK][VNODES], int* syndrome, int size_checks, int size_vnode, float Lj[VNODES], float alpha, int num_it, int pcm_matrix[CHECK][VNODES]){
    
    for(int i = 0; i < num_it; i++){
        
        float sum[VNODES];
        int error[VNODES];

        compute_row_operations(L, syndrome, size_checks, size_vnode);
        printf("L matrix after row ops:\n");
        for(int i = 0; i < CHECK; i++){
            for(int j = 0; j < VNODES; j++){
                printf("%f ", L[i][j]);
            }
            printf("\n");
        }

        compute_col_operations(L, syndrome, size_checks, size_vnode, alpha, Lj,sum);

        // Correct syndrome from the values of the codeword
        // if >= 0 then bit j = 0
        // if < 0  then bit j = 1 
        for(int j = 0; j < VNODES; j++){

            if (sum[j] >= 0) error[j] = 0;
            else error[j] = 1;
        }
        // ----------- DEBUG PRINT --------------
        printf("error computed: ");
        for(int j = 0; j < VNODES; j++){

            printf("%d ",error[j]);
        }
        printf("\n");

        //------------------------------------------
        //compute S = eH^T
        int resulting_syndrome[CHECK];
        for(int i = 0; i < CHECK; i++){
            int row_op = 0;
            for(int j = 0; j < VNODES; j++){
                row_op ^= (error[j] & pcm_matrix[i][j]);
            }
            resulting_syndrome[i] = row_op;
        }
        int error_found = 1;

        // ----------- DEBUG PRINT --------------
        for(int i = 0; i < CHECK; i++){
            printf("resulting syndrome %d \n",resulting_syndrome[i]);
            if(resulting_syndrome[i] != syndrome[i]) error_found = 0;
        }
        
        if(error_found) {
            printf("error found on it %d \n",i);
            break;
        }
        //------------------------------------------
    }
}

void compute_row_operations(float L[CHECK][VNODES], int* syndrome, int size_checks, int size_vnode){

    for(int i = 0; i < CHECK; i++){
        if(i == size_checks) break;
   
        float min1 = FLT_MAX, min2 = FLT_MAX;
        int minpos = -1;
        int sign_minpos = 0;
        int row_sign = 0;
        float product = 1.0;

        // Search min1 and min2
        for(int j = 0; j < VNODES; j++){
            if(j == size_vnode) break;

            float val = L[i][j];
            float abs_val = fabs(val);

            if(val != 0.0f){
                if(abs_val < min1){
                    min2 = min1;
                    min1 = abs_val;
                    minpos = j;
                    sign_minpos = val >= 0 ? 0 : 1;
                }else if (abs_val < min2) {
                    min2 = abs_val;
                }
            }

            row_sign = row_sign ^ (val >= 0 ? 0 : 1);
        }

        if(min2 == FLT_MAX){
            fprintf(stderr, "There's less than two 1's on the row %d", i);
            exit(EXIT_FAILURE);
        }

        // Apply the corresponding value and sign to out[][]
        for(int j = 0; j < VNODES; j++){
            if(j == size_vnode) break;

            float val = L[i][j];

            if(val != 0.0f){
                // sign is negative (-1.0f) if the final signbit (operation in parethesis) is 0, 
                // and positive (1.0f) if its 1
                float sign =  1.0f - 2.0f * (row_sign ^ (val >= 0 ? 0 : 1) ^ syndrome[i]);
                
                // Assign min2 to minpos when loop ends to save if statements
                L[i][j] = sign * min1;
            } else{
                L[i][j] = 0.0f;
            }
        }

        // Assigning min2 to minpos
        L[i][minpos] = (1.0f - 2.0f * (row_sign ^ sign_minpos ^ syndrome[i])) * min2;
    }    
}

void compute_col_operations(float L[CHECK][VNODES], int* syndrome, int size_checks, int size_vnode, float alpha, float Lj[CHECK], float sum[VNODES]){
    
    for (int j = 0; j < VNODES; j++){
        if (j == size_vnode) break;

        // Possible optimization: Read entire column L[][j] to another variable beforehand and then add the values
        float suma_aux = 0.0f;
        for(int i = 0; i < CHECK; i++){
            if (i == size_checks) break;

            suma_aux += L[i][j];
        }

        sum[j] = Lj[j] + alpha * suma_aux;
    }

    for (int i = 0; i < CHECK; i++){
        if(i == size_checks) break;

        for (int j = 0; j < VNODES; j++){
            if(j == size_vnode) break;

            if(L[i][j] != 0) L[i][j] = sum[j] - (alpha * L[i][j]);
        }
    }

}
                            

int main() {
    float L[CHECK][VNODES];
    int pcm_matrix[CHECK][VNODES];
    float Lj[VNODES];
    FILE *file = fopen("input2.txt","r");
    if (file == NULL){
        perror("Error opening file");
        return 1;
    }

    //read the probability p of the error model
    float p;
    if (fscanf(file, "%f", &p) != 1) {
            fprintf(stderr, "Error reading probability p\n");
            fclose(file);
            return 1;
    }
    p = (1.0f -(2.0f/3.0f)*p);


    // Read rows and cols
    int rows,cols;
    if (fscanf(file, "%d %d", &rows, &cols) != 2) {
        fprintf(stderr, "Error reading dimensions\n");
        fclose(file);
        return 1;
    }
    //
    // Read matrix L (initial beliefs)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int value;
            if (fscanf(file, "%d", &value) != 1) {
                fprintf(stderr, "Error reading valor of row %d col %d\n", i, j);
                fclose(file);
                return 1;
            }
            pcm_matrix[i][j] = value;
            if(value == 1){
                L[i][j] = p;
            }else{
                L[i][j] = 0;
            }
        }
    }

    int syndrome[CHECK];
    // Read syndrome
    for (int j = 0; j < rows; j++) {
        if (fscanf(file, "%d", &syndrome[j]) != 1) {
            fprintf(stderr, "Error reading syndrome[%d]\n", j);
            fclose(file);
            return 1;
        }
    }

  
    // Initialize Lj
    for (int i = 0; i < cols; i++) {
        Lj[i] = p;
    }

    // Read alpha
    float alpha;
    if (fscanf(file, "%f", &alpha) != 1) {
        fprintf(stderr, "Error reading alpha\n");
        fclose(file);
        return 1;
    }

    float out[CHECK][VNODES];

    compute_check_to_value(L, syndrome, rows, cols, Lj, alpha,10,pcm_matrix);

   // ----------- DEBUG PRINT --------------
    printf("Matrix L:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.6f ", L[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    // -------------------------------------
    
    return 0;
}