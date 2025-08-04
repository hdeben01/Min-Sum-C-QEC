#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include "compute_check_to_value_msg.h"

/*
COMENTARIOS:

Para comprobar mejor si va todo bien estaría bien crear un caso específico sabiendo el 
resultado, ya que el síndrome que has puesto tú (señor Hugo de Benito), es lo que le 
sale como resultado en el paper, no el inicial (eso sería el codeword pero eso no nos vale en QEC)

Para el clásico en sí el algortimo está bien, porque si quitas la parte del síndrome que afecta en el 
paso 2, el resultado sale como en el paper, falta hacer que se hagan los loops


He añadido el codeword para comprobar como se haría en clásico, luego me base en el paper 
y el síndrome lo cambie en base a si el valor del codeword es positivo o negativo

Ordené un poco la parte de calcular el signo del paso 2

También quité el argumento out[][] y realice los cambios del paso 2 directamente en L, y 
usé esos valores para sacar sum[j] en el paso 3

Ví que no había problema si se usaba más de una función así que lo dividí un poco
para que quede más claro

*/


void compute_row_operations(float L[CHECK][VNODES], int* syndrome, int size_checks, int size_vnode);
void compute_col_operations(float L[CHECK][VNODES], int* syndrome, int size_checks, int size_vnode, int alpha, float codeword[CHECK]);

//size neigh > 0 y size_syn > 0
//esta función dada una matriz L de mensajes calcula los mensajes que reciben los value nodes de los check nodes y por tanto equivalen al mensaje del "check node"
void compute_check_to_value(float L[CHECK][VNODES], int* syndrome, int size_checks, int size_vnode, float codeword[CHECK], float alpha){
    
    compute_row_operations(L, syndrome, size_checks, size_vnode);

    compute_col_operations(L, syndrome, size_checks, size_vnode, alpha, codeword);

    // Correct syndrome from the values of the codeword
    // if >= 0 then bit j = 0
    // if < 0  then bit j = 1 
    for(int j = 0; j < CHECK; j++){

        if (codeword[j] >= 0) syndrome[j] = 0;
        else syndrome[j] = 1;
    }

}

void compute_row_operations(float L[CHECK][VNODES], int* syndrome, int size_checks, int size_vnode){

    for(int i = 0; i < CHECK; i++){
        if(i == size_checks) break;
   
        float min1 = FLT_MAX, min2 = FLT_MAX;
        int minpos = -1;
        int sign_minpos = 0;
        int row_sign = 1;
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
                    sign_minpos = signbit(val);
                }else if (abs_val < min2) {
                    min2 = abs_val;
                }
            }

            row_sign = row_sign ^ signbit(val);
            //(product = product * (signbit(L[i][j]) ^ syndrome[i]); 
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
                float sign = 1.0f - 2.0f * (row_sign ^ signbit(val) ^ syndrome[i]);

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

void compute_col_operations(float L[CHECK][VNODES], int* syndrome, int size_checks, int size_vnode, int alpha, float codeword[CHECK]){
    
    // Esta parte está 100% bien 😭 ya que porfin entendí como iba 🔥

    float sum[CHECK];

    for (int j = 0; j < VNODES; j++){
        if (j == size_vnode) break;

        // Possible optimization: Read entire column L[][j] to another variable beforehand and then add the values
        float suma_aux = 0.0f;
        for(int i = 0; i < CHECK; i++){
            if (i == size_checks) break;

            suma_aux += L[i][j];
        }

        sum[j] = codeword[j] + alpha * suma_aux;
    }

    for (int i = 0; i < CHECK; i++){
        if(i == size_checks) break;

        for (int j = 0; j < VNODES; j++){
            if(j == size_vnode) break;

            if(L[i][j] != 0) L[i][j] = sum[j] - L[i][j];
        }
    }

}
                            

int main() {
    float L[CHECK][VNODES];
    FILE *file = fopen("input.txt","r");
    if (file == NULL){
        perror("Error opening file");
        return 1;
    }


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
            if (fscanf(file, "%f", &L[i][j]) != 1) {
                fprintf(stderr, "Error reading valor of row %d col %d\n", i, j);
                fclose(file);
                return 1;
            }
        }
    }

    int syndrome[CHECK];
    // Read syndrome
    for (int j = 0; j < cols; j++) {
        if (fscanf(file, "%d", &syndrome[j]) != 1) {
            fprintf(stderr, "Error reading syndrome[%d]\n", j);
            fclose(file);
            return 1;
        }
    }

    float codeword[CHECK];
    // Read codeword
    for (int i = 0; i < cols; i++) {
        if (fscanf(file, "%f", &codeword[i]) != 1) {
            fprintf(stderr, "Error reading codeword[%d]\n", i);
            fclose(file);
            return 1;
        }
    }

    // Read alpha
    float alpha;
    if (fscanf(file, "%f", &alpha) != 1) {
        fprintf(stderr, "Error reading alpha\n");
        fclose(file);
        return 1;
    }

    float out[CHECK][VNODES];

    compute_check_to_value(L, syndrome, rows, cols, codeword, alpha);

    // Show Matrix L 
    printf("Matrix L:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.6f ", L[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    // Show corrected syndrome
    printf("Corrected Syndrome:");
    for(int j = 0; j < cols; j++) printf(" %d", syndrome[j]);
    printf("\n");
    
    return 0;
}