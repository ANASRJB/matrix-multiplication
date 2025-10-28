#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

void matrix_multiply_openmp(double *A, double *B, double *C, int size) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            double sum = 0.0;
            for (int k = 0; k < size; k++) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
}

void benchmark_openmp(int max_size, int step) {
    printf("OpenMP Benchmarking...\n");
    
    for (int size = 100; size <= max_size; size += step) {
        // Allocate matrices
        double *A = (double*)malloc(size * size * sizeof(double));
        double *B = (double*)malloc(size * size * sizeof(double));
        double *C = (double*)malloc(size * size * sizeof(double));
        
        // Initialize with random values
        for (int i = 0; i < size * size; i++) {
            A[i] = (double)rand() / RAND_MAX;
            B[i] = (double)rand() / RAND_MAX;
        }
        
        // Time the multiplication
        double start = omp_get_wtime();
        matrix_multiply_openmp(A, B, C, size);
        double end = omp_get_wtime();
        
        printf("Size: %d, Time: %.4f seconds\n", size, end - start);
        
        free(A);
        free(B);
        free(C);
    }
}

int main() {
    srand(time(NULL));
    benchmark_openmp(1000, 100);
    return 0;
}