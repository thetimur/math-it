#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h> 

#define BS 32
#define EPS 0.1
#define NUM_RUNS 10

double **allocate_matrix(int rows, int cols) {
    double **matrix = (double **)malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (double *)malloc(cols * sizeof(double));
    }

    return matrix;
}

void free_matrix(double **matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }

    free(matrix);
}

double f(double x, double y) {
    return 0;
}

double g(double x, double y) {
    enum { X_EQUALS_0 = 0, Y_EQUALS_0, X_EQUALS_1, OTHER_CASE } boundary_case;

    if (x == 0) {
        boundary_case = X_EQUALS_0;
    } else if (y == 0) {
        boundary_case = Y_EQUALS_0;
    } else if (x == 1) {
        boundary_case = X_EQUALS_1;
    } else {
        boundary_case = OTHER_CASE;
    }

    switch (boundary_case) {
        case X_EQUALS_0:
            return 100 - 200 * y;
        case Y_EQUALS_0:
            return 100 - 200 * x;
        case X_EQUALS_1:
            return -100 + 200 * y;
        case OTHER_CASE:
            return -100 + 200 * x;
        default:
            return 0;
    }
}

double recalculate(double **u, int i, int j, double h, double **fm) {
    return 0.25 * fabs(u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1] - h * h * fm[i][j]);
}

double process_block(int i_block, int j_block, int N, double **u, double **fm, double h) {
    int li = i_block * BS + 1;
    int lj = j_block * BS + 1;

    int ri = fmin(li + BS, N);
    int rj = fmin(lj + BS, N);

    double block_max = 0;
    for (int i = li; i < ri; ++i) {
        for (int j = lj; j < rj; ++j) {
            double old_u = u[i][j];

            u[i][j] = recalculate(u, i, j, h, fm);
            block_max = fmax(block_max, fabs(old_u - u[i][j]));
        }
    }
    return block_max;
}

void calculate_and_measure(int N, double **u, double **f) {
    int NB = ceil((double)N / BS);
    double h = 1.0 / (N + 1);

    double *dms = (double *)malloc(NB * sizeof(double));
    double diff = 0;
    do {
        diff = 0;
        for (int nx = 0; nx < NB; ++nx) {
            dms[nx] = 0;
            #pragma omp parallel for shared(nx)
            for (int i = 0; i < nx + 1; ++i) {
                int j = nx - i;
                double d = process_block(i, j, N, u, f, h);
                #pragma omp critical
                dms[i] = fmax(dms[i], d);
            }
        }

        for (int nx = NB - 2; nx >= 0; --nx) {
            #pragma omp parallel for shared(nx)
            for (int i = 1; i < nx + 1; ++i) {
                int j = 2 * (NB - 1) - nx - i;
                double d = process_block(i, j, N, u, f, h);
                #pragma omp critical
                dms[i] = fmax(dms[i], d);
            }
        }
        for (int i = 0; i < NB; ++i) {
            diff = fmax(diff, dms[i]);
        }
    } while (diff > EPS);
    free(dms);
}

int compare_doubles(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

int main() {
    int thread_count[] = {1, 4, 8};
    int NN[] = {100, 200, 300, 500, 1000, 2000, 3000};

    for (int th = 0; th < sizeof(thread_count) / sizeof(thread_count[0]); ++th) {
        omp_set_num_threads(thread_count[th]);

        for (int cur_idx = 0; cur_idx < sizeof(NN) / sizeof(NN[0]); ++cur_idx) {

            int N = NN[cur_idx];
            double h = 1.0 / (N + 1);
            
            double total_time = 0;
            double min_time = 1e9;
            double max_time = 0.0;
            double *times = (double *)malloc(NUM_RUNS * sizeof(double));

            for (int run = 0; run < NUM_RUNS; ++run) {
                double **u = allocate_matrix(N + 2, N + 2);
                double **fm = allocate_matrix(N + 2, N + 2);
                
                for (int i = 1; i < N + 1; ++i) {
                    for (int j = 1; j < N + 1; ++j) {
                        fm[i][j] = f(i * h, j * h);
                    }
                }

                for (int i = 0; i < N + 1; ++i) {
                    u[i][0] = g(i * h, 0);
                    u[0][i + 1] = g(0, (i + 1) * h);
                    u[i + 1][N + 1] = g((i + 1) * h, (N + 1) * h);
                    u[N + 1][i] = g((N + 1) * h, i * h);
                }

                double start_time = omp_get_wtime();
                calculate_and_measure(N, u, fm);
                double end_time = omp_get_wtime();

                double current = end_time - start_time;
                times[run] = current;
                total_time += current;
                min_time = fmin(min_time, current);
                max_time = fmax(max_time, current);

                free_matrix(u, N + 2);
                free_matrix(fm, N + 2);
            }

            qsort(times, NUM_RUNS, sizeof(double), compare_doubles);
            double median_time = NUM_RUNS % 2 == 0 ? (times[NUM_RUNS / 2 - 1] + times[NUM_RUNS / 2]) / 2.0 : times[NUM_RUNS / 2];

            printf("threads = %d, N = %d, average_time = %f, min_time= %f, max_time= %f, median_time = %f\n", thread_count[th], N, total_time / (double)NUM_RUNS, min_time, max_time, median_time);
        }
    }
    return 0;
}
