#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define NUM_THREADS 4

typedef struct {
    double *A, *B, *C;
    int size, start_row, end_row;
} thread_data_t;

void* multiply_thread(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;
    
    for (int i = data->start_row; i < data->end_row; i++) {
        for (int j = 0; j < data->size; j++) {
            double sum = 0.0;
            for (int k = 0; k < data->size; k++) {
                sum += data->A[i * data->size + k] * data->B[k * data->size + j];
            }
            data->C[i * data->size + j] = sum;
        }
    }
    pthread_exit(NULL);
}

void matrix_multiply_pthreads(double *A, double *B, double *C, int size) {
    pthread_t threads[NUM_THREADS];
    thread_data_t thread_data[NUM_THREADS];
    
    int rows_per_thread = size / NUM_THREADS;
    
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].A = A;
        thread_data[i].B = B;
        thread_data[i].C = C;
        thread_data[i].size = size;
        thread_data[i].start_row = i * rows_per_thread;
        thread_data[i].end_row = (i == NUM_THREADS - 1) ? size : (i + 1) * rows_per_thread;
        
        pthread_create(&threads[i], NULL, multiply_thread, &thread_data[i]);
    }
    
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
}

void benchmark_pthreads(int max_size, int step) {
    printf("Pthreads Benchmarking...\n");
    
    for (int size = 100; size <= max_size; size += step) {
        double *A = (double*)malloc(size * size * sizeof(double));
        double *B = (double*)malloc(size * size * sizeof(double));
        double *C = (double*)malloc(size * size * sizeof(double));
        
        for (int i = 0; i < size * size; i++) {
            A[i] = (double)rand() / RAND_MAX;
            B[i] = (double)rand() / RAND_MAX;
        }
        
        clock_t start = clock();
        matrix_multiply_pthreads(A, B, C, size);
        clock_t end = clock();
        
        double time_spent = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("Size: %d, Time: %.4f seconds\n", size, time_spent);
        
        free(A);
        free(B);
        free(C);
    }
}

int main() {
    srand(time(NULL));
    benchmark_pthreads(1000, 100);
    return 0;
}