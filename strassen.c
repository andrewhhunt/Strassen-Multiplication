/*
    Strassen algorithm parallellized with OpenMP
    Andrew Hunt

    Multiplies two large matrices using the Strassen algorithm and OpenMP

    File type needs to be .c, the website would only allow uploading txt files
    Needs the -fopenmp flag when compiling
    gcc -fopenmp strassen.c

    usage: strassen.exe -m 10 -k 8 -t 8 -c
    -m sets power of 2 for matrix size
    -k sets smallest matrix size to multiply normally
    -t number of threads
    -c checks results with normal matrix multiplication
    -h explains the flags
*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*

additions:
a1 = a11 + a22
a2 = b11 + b22
a3 = a21 + a22
a4 = b12 - b22
a5 = b21 - b11
a6 = a11 + a12
a7 = a21 - a11
a8 = b11 + b12
a9 = a12 - a22
a10 = b21 + b22

multiplications:
m1 = a1 * a2
m2 = a3 * b11
m3 = a11 * a4
m4 = a22 * a5
m5 = a6 * b22
m6 = a7 * a8
m7 = a9 * a10

c11 = M1 + M4 - M5 + m7
c12 = M3 + m5
c21 = M2 + M4
c22 = M1 - M2 + M3 + M6

*/

int ** a;
int ** b;
int ** c;

int kp;

/**
 * Frees the memory associated with the provided matrix
 * 
 * mat The matrix to be freed
 * size The size of the matrix
 */
void free_matrix(int ** mat, int size) {
    for (int i = 0; i < size; i++) {
        free(mat[i]);
    }

    free(mat);
}

/**
 * Create a square matrix of the provided size
 * 
 * size Size of the desired matrix
 * int** Pointer to the created matrix
 */
int ** create_matrix(int size) {
    int ** mat = malloc(size * sizeof(int *));

    for (int i = 0; i < size; i++) {
        mat[i] = malloc(size * sizeof(int));
    }

    return mat;
}

/**
 * Prints a square matrix
 * 
 * mat The matrix to be printed
 * size The size of the matrix
 */
void print_matrix(int ** mat, int size) {
    for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    printf("%d\t", mat[i][j]);
                }
                printf("\n");
            }
}


/**
 *  Adds two square matrices and saves them to the result matrix.
 * 
 * left + right = result
 * 
 * left The left matrix 
 * right The right matrix
 * result The result matrix
 * size The size of the matrices
 * subtract Set to 1 to subtract, 0 to add
 */
void matrix_add(int ** left, int ** right, int ** result, int size, int subtract) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (subtract) {
                result[i][j] = left[i][j] - right[i][j];
            } else {
                result[i][j] = left[i][j] + right[i][j];
            } 
        }
    }
}


/**
 * Copies the source matrix into a quadrant of the destination matrix
 * 
 * source The source matrix
 * dest The larger destination matrix
 * source_size The size of the source matrix
 * dest_i 0 if the destination is on the left, 1 if on the right
 * dest_j 0 if the destination is on the top, 1 if on the bottom
 */
void matrix_copy_into(int ** source, int ** dest, int source_size, int dest_i, int dest_j) {
    for (int i = 0; i < source_size; i++) {
        for (int j = 0; j < source_size; j++) {
            dest[i+(dest_i*source_size)][j+(dest_j*source_size)] = source[i][j];
        }
    }
}


/**
 * Copies from a quadrant of the source matrix into the destination matrix
 * 
 * source The larger source matrix
 * dest The destination matrix
 * dest_size The size of the destination matrix
 * source_i 0 if the source is on the left, 1 if on the right
 * source_j 0 if the source is on the top, 1 if on the bottom
 */
void matrix_copy_from(int ** source, int ** dest, int dest_size, int source_i, int source_j) {
    for (int i = 0; i < dest_size; i++) {
        for (int j = 0; j < dest_size; j++) {
            dest[i][j] = source[i+(source_i*dest_size)][j+(source_j*dest_size)];
        }
    }
}


/**
 * Performs a recursive, parallel implementation of the Strassen matrix multiplication
 * algorithm. left * right = prod
 * 
 * left The left square matrix
 * right The right square matrix
 * prod The product of the two square matrices
 * size The size of the matrices
 */
void strassen(int ** left, int ** right, int ** prod, int size) {
    if (size == 1) {
        // The inputs are 1, return the two multiplied together
        prod[0][0] = left[0][0] * right[0][0];
    } else if (size == kp) {
        // The size has reached k', so just do normal matrix multiplication
        for (int i = 0; i < size; i++) {
             for (int j = 0; j < size; j++) {
                prod[i][j] = 0;
                for (int k = 0; k < size; k++) {
                    prod[i][j] += left[i][k] * right[k][j];
                }
             }
        }
    } else {
        // split the matrix into quadrants
        int split_size = size / 2;

        // matrix quadrants
        int ** a11 = create_matrix(split_size);
        int ** a12 = create_matrix(split_size);
        int ** a21 = create_matrix(split_size);
        int ** a22 = create_matrix(split_size);

        int ** b11 = create_matrix(split_size);
        int ** b12 = create_matrix(split_size);
        int ** b21 = create_matrix(split_size);
        int ** b22 = create_matrix(split_size);

        // addition results
        int ** add1 = create_matrix(split_size);
        int ** add2 = create_matrix(split_size);
        int ** add3 = create_matrix(split_size);
        int ** add4 = create_matrix(split_size);
        int ** add5 = create_matrix(split_size);
        int ** add6 = create_matrix(split_size);
        int ** add7 = create_matrix(split_size);
        int ** add8 = create_matrix(split_size);
        int ** add9 = create_matrix(split_size);
        int ** add10 = create_matrix(split_size);
        
        // 7 products
        int ** p1 = create_matrix(split_size);
        int ** p2 = create_matrix(split_size);
        int ** p3 = create_matrix(split_size);
        int ** p4 = create_matrix(split_size);
        int ** p5 = create_matrix(split_size);
        int ** p6 = create_matrix(split_size);
        int ** p7 = create_matrix(split_size);
        
        
        // Copy data into matrices
        #pragma omp parallel
        {
            matrix_copy_from(left, a11, split_size, 0, 0);
            matrix_copy_from(left, a12, split_size, 0, 1);
            matrix_copy_from(left, a21, split_size, 1, 0);
            matrix_copy_from(left, a22, split_size, 1, 1);
            matrix_copy_from(right, b11, split_size, 0, 0);
            matrix_copy_from(right, b12, split_size, 0, 1);
            matrix_copy_from(right, b21, split_size, 1, 0);
            matrix_copy_from(right, b22, split_size, 1, 1);
        }
        

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                // P1
                matrix_add(a11, a22, add1, split_size, 0);  // A1 = a11 + a22
                matrix_add(b11, b22, add2, split_size, 0);  // A2 = b11 + b22
                strassen(add1, add2, p1, split_size);       // P1 = A1 * A2
            }

            #pragma omp section
            {
                // P2
                matrix_add(a21, a22, add3, split_size, 0);  // A3 = a21 + a22
                strassen(add3, b11, p2, split_size);        // P2 = A3 * b11
            }

            #pragma omp section
            {
                // P3
                matrix_add(b12, b22, add4, split_size, 1);  // A4 = b12 - b22
                strassen(a11, add4, p3, split_size);        // P3 = a11 * A4
            }

            #pragma omp section
            {
                // P4
                matrix_add(b21, b11, add5, split_size, 1);  // A5 = b21 - b11
                strassen(a22, add5, p4, split_size);        // P4 = a22 * A5
            }

            #pragma omp section
            {
                // P5
                matrix_add(a11, a12, add6, split_size, 0);  // A6 = a11 + a12
                strassen(add6, b22, p5, split_size);        // P5 = A6 * b22
            }

            #pragma omp section
            {
                // P6
                matrix_add(a21, a11, add7, split_size, 1);  // A7 = a21 - a11
                matrix_add(b11, b12, add8, split_size, 0);  // A8 = b11 + b12
                strassen(add7, add8, p6, split_size);       // P6 = A7 * A8 
            }

            #pragma omp section
            {
                // P7
                matrix_add(a12, a22, add9, split_size, 1);  // A9 = a12 - a22
                matrix_add(b21, b22, add10, split_size, 0); // A10 = b21 + b22
                strassen(add9, add10, p7, split_size);      // P7 = A9 * A10
            }
        }
        
        #pragma omp parallel
        {
            // add up products for A11
            matrix_add(p1, p4, add1, split_size, 0);        // (p1 + p4)
            matrix_add(p5, p7, add2, split_size, 1);        // (p5 - p7)
            matrix_add(add1, add2, add3, split_size, 1);    // (p1 + p4) - (p5 - p7)
            matrix_copy_into(add3, prod, split_size, 0, 0);
         
            // add up products for A12
            matrix_add(p3, p5, add4, split_size, 0);        // (p3 + p5)
            matrix_copy_into(add4, prod, split_size, 0, 1);
         
            // add up products for A21
            matrix_add(p2, p4, add5, split_size, 0);        // (p2 + p4)
            matrix_copy_into(add5, prod, split_size, 1, 0);
         
            // Add up products for A22
            matrix_add(p1, p2, add6, split_size, 1);        // (p1 - p2)
            matrix_add(p3, p6, add7, split_size, 0);        // (p3 + p6)
            matrix_add(add6, add7, add8, split_size, 0);    // (p1 - p2) + (p3 + p6)   
            matrix_copy_into(add8, prod, split_size, 1, 1);
        }

        // Free memory
        free_matrix(a11, split_size);
        free_matrix(a12, split_size);
        free_matrix(a21, split_size);
        free_matrix(a22, split_size);
        free_matrix(b11, split_size);
        free_matrix(b12, split_size);
        free_matrix(b21, split_size);
        free_matrix(b22, split_size);
        free_matrix(add1, split_size);
        free_matrix(add2, split_size);
        free_matrix(add3, split_size);
        free_matrix(add4, split_size);
        free_matrix(add5, split_size);
        free_matrix(add6, split_size);
        free_matrix(add7, split_size);
        free_matrix(add8, split_size);
        free_matrix(add9, split_size);
        free_matrix(add10, split_size);
        free_matrix(p1, split_size);
        free_matrix(p2, split_size);
        free_matrix(p3, split_size);
        free_matrix(p4, split_size);
        free_matrix(p5, split_size);
        free_matrix(p6, split_size);
        free_matrix(p7, split_size);
    }

}


int main(int argc, char *argv[]) {
    double total_time;
    double time_start, time_stop;

    int matrix_size, threads;
    int check = 0;
    int print = 0;

    int m = -1;
    int k = -1;
    int t = -1;

    // Check terminal options
    for (int i = 1; i < argc; i++) {
        if (strcmp("-m", argv[i]) == 0) { // Matrix size
            m = atoi(argv[++i]);
            if (0 < m & m < 31) {
                matrix_size = (1 << m);
                printf("Matrix size set to: %d\n", matrix_size);
            } else {
                printf("Invalid matrix size: %d\n", m);
                return -1;
            }
        } else if (strcmp("-k",argv[i]) == 0) {  // k' limit
            k = atoi(argv[++i]);
            if (0 < k & k < 15) {
                kp = (1 << k);
                printf("k' set to %d\n", kp);
            } else {
                printf("Invalid k' value\n");
                return -1;
            }
        } else if(strcmp("-t",argv[i]) == 0) { // Number of threads
            t = atoi(argv[++i]);
            if (0 < t & t < 49) {
                threads = t;
                printf("Number of threads set to: %d\n", t);
            } else {
                printf("Invalid number of threads: %d\n", t);
                return -1;
            }
        } else if (strcmp("-c",argv[i]) == 0) { // Check
            check = 1;
            printf("Matrix checking enabled\n");
        } else if (strcmp("-p",argv[i]) == 0) { // Print
            print = 1;
            printf("Printing enabled\n");
        } else if (strcmp("-h",argv[i]) == 0) { // Help
            printf("Available options:\n-m <2^m> : size of matrices as a power of 2\n-k <2^k> : Size of matrix to stop using strassen as a power of 2\n-t <# threads> : Number of threads to use\n-c : Check with normal multiplication\n-p : Print the matrices");
            return 0;
        } else {
            printf("Option not found: %s\n", argv[i]);
        }
    }

    // Check if m was set
    if (m == -1) {
        printf("No matrix size set.\nCorrect usage: strassen.exe -m <2^m>\n");
        return -1;
    }

    // Check if t was set
    if (t == -1) {
        printf("No thread number set, using 1 thread.\nCorrect usage: strassen.exe -t <# threads>\n");
        threads = 1;
    }

    // Check if k was set
    if (k == -1) {
        kp = 1;
    }

    // Set the number of threads to use
    omp_set_num_threads(threads);

    //srand(343);

    // Create the left, right, and product matrices
    a = create_matrix(matrix_size);
    b = create_matrix(matrix_size);
    c = create_matrix(matrix_size);

    // Set the values for the matrices
    for(int i = 0; i < matrix_size; i++) {
        for(int j = 0; j < matrix_size; j++) {
            a[i][j] = rand() % 1000;
            b[i][j] = rand() % 1000;
            c[i][j] = 0;
        }
    }

    if (print) {
        // If the print flag is set, print out the starting matrices
        printf("A:\n");
        print_matrix(a, matrix_size);

        printf("-------\n");
        printf("B:\n");
        print_matrix(b, matrix_size);
        printf("-------\n");
    }
    
    printf("Calculating with Strassen\n");

    time_start = omp_get_wtime();
    
    strassen(a, b, c, matrix_size);
    
    time_stop = omp_get_wtime();
    total_time = time_stop - time_start;
    
    printf("Strassen:\t%f seconds\n", total_time);
    

    if (check) {
        // If the check flag is set, create another matrix and multiply normally
        printf("Calculating normal:\n");

        int ** d = create_matrix(matrix_size);

        time_start = omp_get_wtime();
        for(int i = 0; i < matrix_size; i++) {
            for(int j = 0; j < matrix_size; j++) {
                d[i][j] = 0;
                for (int k = 0; k < matrix_size; k++) {
                    d[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        time_stop = omp_get_wtime();
        total_time = time_stop - time_start;
        printf("Normal:\t\t%f seconds\n", total_time);
        

        int error = 0;

        // Compare the values of each product matrix
        for (int i = 0; i < matrix_size; i++) {
            for (int j = 0; j < matrix_size; j++) {
                if (c[i][j] != d[i][j]) {
                    error = 1;
                    break;
                }
            }
        }

        if (error == 0) {
            printf("Everything worked!\n");
        } else {
            printf("NOTHING WORKED!\n");
        }

        if (print) {
            print_matrix(c, matrix_size);

            printf("-------\n");

            print_matrix(d, matrix_size);
        }

        free_matrix(d, matrix_size);
    }

    // Free memory
    free_matrix(a, matrix_size);
    free_matrix(b, matrix_size);
    free_matrix(c, matrix_size);

    return 0;
}