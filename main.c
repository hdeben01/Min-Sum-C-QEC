#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include "compute_check_to_value_msg.h"


void compute_row_operations(float L[CHECK][VNODES], const int non_zero[CHECK][VNODES],
                            int* syndrome, int size_checks, int size_vnode);

void compute_col_operations(float L[CHECK][VNODES], const int non_zero[CHECK][VNODES], 
                            int* syndrome, int size_checks, int size_vnode, float alpha, 
                            float Lj[VNODES], float sum[VNODES]);

void show_matrix(const float matrix[CHECK][VNODES], const int non_zero[CHECK][VNODES],
                 const int rows, const int cols);

//size neigh > 0 y size_syn > 0
//esta función dada una matriz L de mensajes calcula los mensajes que reciben
//los value nodes de los check nodes y por tanto equivalen al mensaje del "check node"
void compute_check_to_value(float L[CHECK][VNODES], const int pcm_matrix[CHECK][VNODES], 
                            int* syndrome, int size_checks, int size_vnode, 
                            float Lj[VNODES], float alpha, int num_it)
{

    for(int i = 0; i < num_it; i++){

        color_printf(CYAN, "Iteration %d\n", i+1);


        float sum[VNODES];
        int error[VNODES];

        compute_row_operations(L, pcm_matrix, syndrome, size_checks, size_vnode);
        printf("\tL matrix after row ops:\n");
        show_matrix(L, pcm_matrix, size_checks, size_vnode);

        compute_col_operations(L, pcm_matrix, syndrome, size_checks, size_vnode, alpha, Lj, sum);
        printf("\tL matrix after col ops:\n");
        show_matrix(L, pcm_matrix, size_checks, size_vnode);


        // Correct syndrome from the values of the codeword
        // if >= 0 then bit j = 0
        // if < 0  then bit j = 1 
        for(int j = 0; j < VNODES; j++){
            if (sum[j] >= 0) error[j] = 0;
            else error[j] = 1;
        }

        // ----------- DEBUG PRINT --------------
        printf("\tError computed: ");
        for(int j = 0; j < VNODES; j++){
            printf("%d ",error[j]);
        }
        printf("\n");
        //------------------------------------------

        // Compute S = eH^T
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
        printf("\tResulting syndrome: ");
        for(int i = 0; i < CHECK; i++){
            printf("%d ",resulting_syndrome[i]);
            if(resulting_syndrome[i] != syndrome[i]) error_found = 0;
        }
        printf("\n");

        if(error_found) {
            color_printf(GREEN, "\tERROR FOUND\n");
            break;
        }
        else if (i == num_it - 1) color_printf(RED, "\nUSED ALL ITERATIONS WITHOUT FINDING THE ERROR");

        printf("\n");
        //------------------------------------------
    }
}

void compute_row_operations(float L[CHECK][VNODES], const int non_zero[CHECK][VNODES], 
                            int* syndrome, int size_checks, int size_vnode)
{

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

            if(non_zero[i][j]){
                if(abs_val < min1){
                    min2 = min1;
                    min1 = abs_val;
                    minpos = j;
                    sign_minpos = (val >= 0 ? 0 : 1);
                }else if (abs_val < min2) {
                    min2 = abs_val;
                }
            }

            row_sign = row_sign ^ (val >= 0 ? 0 : 1);
        }

        // Apply the corresponding value and sign to out[][]
        for(int j = 0; j < VNODES; j++){
            if(j == size_vnode) break;

            float val = L[i][j];

            if(non_zero[i][j]){
                // sign is negative (-1.0f) if the final signbit (operation in parethesis) is 0, 
                // and positive (1.0f) if its 1
                float sign =  1.0f - (2.0f * (row_sign ^ (val >= 0 ? 0 : 1) ^ syndrome[i]));

                // Assign min2 to minpos when loop ends to save if statements
                L[i][j] = sign * min1;
            }
        }

        // Assigning min2 to minpos
        L[i][minpos] = (1.0f - 2.0f * (row_sign ^ sign_minpos ^ syndrome[i])) * min2;
    }
}

void compute_col_operations(float L[CHECK][VNODES], const int non_zero[CHECK][VNODES],
                            int* syndrome, int size_checks, int size_vnode, float alpha, 
                            float Lj[CHECK], float sum[VNODES])
{
    
    for (int j = 0; j < VNODES; j++){
        if (j == size_vnode) break;

        // Possible optimization: Read entire column L[][j] to another variable beforehand and then add the values
        float sum_aux = 0.0f;
        for(int i = 0; i < CHECK; i++){
            if (i == size_checks) break;

            sum_aux += L[i][j];
        }

        sum[j] = Lj[j] + (alpha * sum_aux);
    }

    for (int i = 0; i < CHECK; i++){
        if(i == size_checks) break;

        for (int j = 0; j < VNODES; j++){
            if(j == size_vnode) break;

            if(non_zero[i][j]) L[i][j] = sum[j] - (alpha * L[i][j]);
        }
    }

}

int main() {
    float L[CHECK][VNODES];
    int pcm_matrix[CHECK][VNODES];
    float Lj[VNODES];
    FILE *file = fopen("input3.txt","r");
    if (file == NULL){
        perror("Error opening file");
        return 1;
    }

    // Read the probability p of the error model
    float p;
    if (fscanf(file, "%f", &p) != 1) {
            fprintf(stderr, "Error reading probability p\n");
            fclose(file);
            return 1;
    }
    p = (1.0f - (2.0f/3.0f) * p);


    // Read rows and cols
    int rows,cols;
    if (fscanf(file, "%d %d", &rows, &cols) != 2) {
        fprintf(stderr, "Error reading dimensions\n");
        fclose(file);
        return 1;
    }

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

    // Read syndrome
    int syndrome[CHECK];
    for (int j = 0; j < rows; j++) {
        if (fscanf(file, "%d", &syndrome[j]) != 1) {
            fprintf(stderr, "Error reading syndrome[%d]\n", j);
            fclose(file);
            return 1;
        }
    }
    printf("Initial Syndrome:");
    for(int i = 0; i < CHECK; i++) printf(" %d", syndrome[i]);
    printf("\n");

    // Initialize Lj
    for (int j = 0; j < cols; j++) {
        Lj[j] = p;
    }
    printf("Lj: %.2f\n", p);

    // Read alpha
    float alpha;
    if (fscanf(file, "%f", &alpha) != 1) {
        fprintf(stderr, "Error reading alpha\n");
        fclose(file);
        return 1;
    }
    printf("Alpha: %.2f\n", alpha);

    int num_it = 10;
    if (fscanf(file, "%d", &num_it) != 1) {
        fprintf(stderr, "Error reading alpha\n");
        fclose(file);
        return 1;
    }
    printf("Max Iterations: %d\n\n", num_it);

    printf("Initial L Matrix:\n");
    show_matrix(L, pcm_matrix, rows, cols);

    compute_check_to_value(L, pcm_matrix, syndrome, rows, cols, Lj, alpha, num_it);
    
    return 0;
}

void show_matrix(const float matrix[CHECK][VNODES], const int non_zero[CHECK][VNODES],
                 const int rows, const int cols)
{
    for(int i = 0; i < rows; i++){
        printf("\t");
        for(int j = 0; j < cols; j++){
            if(signbit(matrix[i][j])) {
                if (non_zero[i][j]) color_printf(YELLOW, " %.6f", matrix[i][j]);
                else printf(" %.6f", matrix[i][j]);
            }
            else {
                if (non_zero[i][j]) color_printf(YELLOW, "  %.6f", matrix[i][j]);
                else printf("  %.6f", matrix[i][j]);
            }
        }
        printf("\n");
    }
    printf("\n");
}
