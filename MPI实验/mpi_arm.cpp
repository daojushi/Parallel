#include <iostream>
#include <mpi.h>
#include <fstream>
#include <cmath>
#include <sys/time.h>
#include <arm_neon.h>
#include <omp.h>
#include<cstring>
using namespace std;

int NUM_THREADS =8;


void init(float** a,int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i][j] = 0;
        }
        a[i][i] = 1.0;
        for (int j = i + 1; j < N; j++) {
            a[i][j] = rand();
        }
    }
    for (int k = 0; k < N; k++) {
        for (int i = k + 1; i < N; i++) {
            for (int j = 0; j < N; j++) {
                a[i][j] += a[k][j];
            }
        }
    }
}
//平凡算法
void normal(float** a,int N) {
    for (int k = 0; k < N; k++) {
        for (int j = k + 1; j < N; j++)
            a[k][j] /= a[k][k];
        a[k][k] = 1.0;
        for (int i = k + 1; i < N; i++) {
            for (int j = k + 1; j < N; j++)
                a[i][j] = a[i][j] - a[i][k] * a[k][j];
            a[i][k] = 0;
        }
    }
}//一维行块划分
double calculate_MPI_block(float**matrix,int N) {
    double start_time, end_time;

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // 只有是0号进程，才进行初始化工作
    if (rank == 0) {
        init(matrix,N);
    }
    start_time = MPI_Wtime();
    int task_num = N / size;
    int remainder = N % size;
    int start = rank * task_num + (rank < remainder ? rank : remainder);
    int end = start + task_num + (rank < remainder ? 1 : 0);
    if (rank == size - 1) {
        end = N;
    }

    // 0号进程负责任务的初始分发工作
    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            int recipient_start = i * task_num + (i < remainder ? i : remainder);
            int recipient_end = recipient_start + task_num + (i < remainder ? 1 : 0);
            if (i == size - 1) {
                recipient_end = N;
            }
            MPI_Send(&matrix[recipient_start][0], (recipient_end - recipient_start) * N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
        }
    }
    // 非0号进程负责任务的接收工作
    else {
        MPI_Recv(&matrix[start][0], (end - start) * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // 做消元运算
    for (int k = 0; k < N; k++) {
        // 如果除法操作是本进程负责的任务，并将除法结果广播
        if (k >= start && k < end) {
            for (int j = k + 1; j < N; j++) {
                matrix[k][j] /= matrix[k][k];
            }
            matrix[k][k] = 1;
            for (int p = 0; p < size; p++) {
                if (p != rank) {
                    MPI_Send(&matrix[k][0], N, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
                }
            }
        }
        // 其余进程接收除法行的结果
        else {
            MPI_Recv(&matrix[k][0], N, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // 进行消元操作
        for (int i = k + 1; i < end; i++) {
            for (int j = k + 1; j < N; j++) {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }

    end_time = MPI_Wtime();
    return (end_time - start_time) * 1000;
}
//行块划分+负载均衡
double calculate_MPI_block_average(float**matrix,int N) {
    double start_time, end_time;

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // 只有是0号进程，才进行初始化工作
    if (rank == 0) {
        init(matrix,N);
    }
    start_time = MPI_Wtime();
    int task_num = N / size;
    int remainder = N % size;
    int start = rank * task_num + (rank < remainder ? rank : remainder);
    int end = start + task_num + (rank < remainder ? 1 : 0);
    if (rank == size - 1) {
        end = N;
    }

    // 0号进程负责任务的初始分发工作
    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            int recipient_start = i * task_num + (i < remainder ? i : remainder);
            int recipient_end = recipient_start + task_num + (i < remainder ? 1 : 0);
            if (i == size - 1) {
                recipient_end = N;
            }
            MPI_Send(&matrix[recipient_start][0], (recipient_end - recipient_start) * N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
        }
    }
    // 非0号进程负责任务的接收工作
    else {
        MPI_Recv(&matrix[start][0], (end - start) * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // 做消元运算
    for (int k = 0; k < N; k++) {
        // 如果除法操作是本进程负责的任务，并将除法结果广播
        if (k >= start && k < end) {
            for (int j = k + 1; j < N; j++) {
                matrix[k][j] /= matrix[k][k];
            }
            matrix[k][k] = 1;
            for (int p = 0; p < size; p++) {
                if (p != rank) {
                    MPI_Send(&matrix[k][0], N, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
                }
            }
        }
        // 其余进程接收除法行的结果
        else {
            MPI_Recv(&matrix[k][0], N, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // 进行消元操作
        for (int i = k + 1; i < end; i++) {
            for (int j = k + 1; j < N; j++) {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }

    end_time = MPI_Wtime();
    return (end_time - start_time) * 1000;
}



// 循环划分并行算法
double calculate_MPI_cycle(float**matrix,int N) {
    double start_time, end_time;

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // 只有是0号进程，才进行初始化工作
    if (rank == 0) {
        init(matrix,N);
    }
    start_time = MPI_Wtime();
    int task_num = rank < N % size ? N / size + 1 : N / size;
    // 0号进程负责任务的初始分发工作
    auto *buff = new float[task_num * N];
    if (rank == 0) {
        for (int p = 1; p < size; p++) {
            for (int i = p; i < N; i += size) {
                for (int j = 0; j < N; j++) {
                    buff[i / size * N + j] = matrix[i][j];
                }
            }
            int count = p < N % size ? N / size + 1 : N / size;
            MPI_Send(buff, count * N, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
        }
    }
        // 非0号进程负责任务的接收工作
    else {
        MPI_Recv(&matrix[rank][0], task_num * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < task_num; i++) {
            for (int j = 0; j < N; j++) {
                matrix[rank + i * size][j] = matrix[rank + i][j];
            }
        }
    }
    // 做消元运算
    for (int k = 0; k < N; k++) {
        // 如果除法操作是本进程负责的任务，并将除法结果广播
        if (k % size == rank) {
            for (int j = k + 1; j < N; j++) {
                matrix[k][j] /= matrix[k][k];
            }
            matrix[k][k] = 1;
            for (int p = 0; p < size; p++) {
                if (p != rank) {
                    MPI_Send(&matrix[k][0], N, MPI_FLOAT, p, 1, MPI_COMM_WORLD);
                }
            }
        }
            // 其余进程接收除法行的结果
        else {
            MPI_Recv(&matrix[k][0], N, MPI_FLOAT, k % size, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // 进行消元操作
        int begin = N / size * size + rank < N ? N / size * size + rank : N / size * size + rank - size;
        for (int i = begin; i > k; i -= size) {
            for (int j = k + 1; j < N; j++) {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
    end_time = MPI_Wtime();
    return (end_time - start_time) * 1000;
}
//一维列块划分
double calculate_MPI_col_block(float** matrix, int N) {
    double start_time, end_time;

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        init(matrix, N);
    }
    start_time = MPI_Wtime();
    int task_num = N / size;
    int remainder = N % size;
    int start = rank * task_num + (rank < remainder ? rank : remainder);
    int end = start + task_num + (rank < remainder ? 1 : 0);
    if (rank == size - 1) {
        end = N;
    }

    // 0号进程负责任务的初始分发工作
    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            int recipient_start = i * task_num + (i < remainder ? i : remainder);
            int recipient_end = recipient_start + task_num + (i < remainder ? 1 : 0);
            if (i == size - 1) {
                recipient_end = N;
            }
            // 将对应列数据分发给各进程
            for (int j = recipient_start; j < recipient_end; j++) {
                MPI_Send(&matrix[0][j], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
        }
    }
    // 非0号进程负责任务的接收工作
    else {
        // 接收对应列的数据
        for (int j = start; j < end; j++) {
            MPI_Recv(&matrix[0][j], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // 做除法运算
    for (int k = 0; k < N; k++) {
        // 如果除法操作是本进程负责的任务，并将除法结果广播
        if (k >= start && k < end) {
            for (int i = k + 1; i < N; i++) {
                matrix[i][k] /= matrix[k][k];
            }
            matrix[k][k] = 1;
            for (int p = 0; p < size; p++) {
                if (p != rank) {
                    MPI_Send(&matrix[0][k], N, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
                }
            }
        }
        // 其余进程接收除法列的结果
        else {
            MPI_Recv(&matrix[0][k], N, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // 进行消元操作
        for (int i = start; i < end; i++) {
            for (int j = k + 1; j < N; j++) {
                matrix[i][j] -= matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }

    end_time = MPI_Wtime();
    return (end_time - start_time) * 1000;
}
double calculate_MPI_SIMD(float**matrix,int N) {
    double start_time, end_time;

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        init(matrix,N);
    }
    start_time = MPI_Wtime();

    int task_num = rank < N% size ? N / size + 1 : N / size;
    auto* buff = new float[task_num * N];

    if (rank == 0) {
        for (int p = 1; p < size; p++) {
            for (int i = p; i < N; i += size) {
                for (int j = 0; j < N; j++) {
                    buff[i / size * N + j] = matrix[i][j];
                }
            }
            int count = p < N% size ? N / size + 1 : N / size;
            MPI_Send(buff, count * N, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
        }
    }
    else {
        MPI_Recv(&matrix[rank][0], task_num * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < task_num; i++) {
            for (int j = 0; j < N; j++) {
                matrix[rank + i * size][j] = matrix[rank + i][j];
            }
        }
    }

    for (int k = 0; k < N; k++) {
        if (k % size == rank) {
            float32x4_t Akk = vmovq_n_f32(matrix[k][k]);
            int j;
            for (j = k + 1; j + 3 < N; j += 4) {
                float32x4_t Akj = vld1q_f32(matrix[k] + j);
                Akj = vdivq_f32(Akj, Akk);
                vst1q_f32(matrix[k] + j, Akj);
            }
            for (; j < N; j++) {
                matrix[k][j] /= matrix[k][k];
            }
        }

        MPI_Bcast(&matrix[k][0], N, MPI_FLOAT, k % size, MPI_COMM_WORLD);

        int begin = N / size * size + rank < N ? N / size * size + rank : N / size * size + rank - size;
        for (int i = begin; i > k; i -= size) {
            float32x4_t Aik = vmovq_n_f32(matrix[i][k]);
            int j;
            for (j = k + 1; j + 3 < N; j += 4) {
                float32x4_t Akj = vld1q_f32(matrix[k] + j);
                float32x4_t Aij = vld1q_f32(matrix[i] + j);
                float32x4_t AikMulAkj = vmulq_f32(Aik, Akj);
                Aij = vsubq_f32(Aij, AikMulAkj);
                vst1q_f32(matrix[i] + j, Aij);
            }
            for (; j < N; j++) {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }

    end_time = MPI_Wtime();
    return (end_time - start_time) * 1000;
}


double calculate_MPI_OMP(float**matrix,int N) {
    double start_time, end_time;

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // 只有是0号进程，才进行初始化工作
    if (rank == 0) {
        init(matrix,N);
    }
    start_time = MPI_Wtime();
    int task_num = rank < N% size ? N / size + 1 : N / size;
    // 0号进程负责任务的初始分发工作
    float* buff = new float[task_num * N];
    if (rank == 0) {
        for (int p = 1; p < size; p++) {
            for (int i = p; i < N; i += size) {
                for (int j = 0; j < N; j++) {
                    buff[i / size * N + j] = matrix[i][j];
                }
            }
            int count = p < N% size ? N / size + 1 : N / size;
            MPI_Send(buff, count * N, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
        }
    }
    // 非0号进程负责任务的接收工作
    else {
        MPI_Recv(&matrix[rank][0], task_num * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < task_num; i++) {
            for (int j = 0; j < N; j++) {
                matrix[rank + i * size][j] = matrix[rank + i][j];
            }
        }
    }
    // 做消元运算
    int i, j, k;
#pragma omp parallel num_threads(NUM_THREADS) default(none) private(i, j, k) shared(matrix, N, size, rank)
    for (k = 0; k < N; k++) {
        // 如果除法操作是本进程负责的任务，并将除法结果广播
#pragma omp single
        {
            if (k % size == rank) {
                for (j = k + 1; j < N; j++) {
                    matrix[k][j] /= matrix[k][k];
                }
                matrix[k][k] = 1;
                for (int p = 0; p < size; p++) {
                    if (p != rank) {
                        MPI_Send(&matrix[k][0], N, MPI_FLOAT, p, 1, MPI_COMM_WORLD);
                    }
                }
            }
            // 其余进程接收除法行的结果
            else {
                MPI_Recv(&matrix[k][0], N, MPI_FLOAT, k % size, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        // 进行消元操作
        int begin = N / size * size + rank < N ? N / size * size + rank : N / size * size + rank - size;
#pragma omp for schedule(guided)
        for (i = begin; i > k; i -= size) {
            for (j = k + 1; j < N; j++) {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
    end_time = MPI_Wtime();
    return (end_time - start_time) * 1000;
}


double calculate_MPI_OMP_SIMD(float**matrix,int N) {
    double start_time, end_time;

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // 只有是0号进程，才进行初始化工作
    if (rank == 0) {
        init(matrix,N);
    }
    start_time = MPI_Wtime();
    int task_num = rank < N% size ? N / size + 1 : N / size;
    // 0号进程负责任务的初始分发工作
    float* buff = new float[task_num * N];
    if (rank == 0) {
        for (int p = 1; p < size; p++) {
            for (int i = p; i < N; i += size) {
                for (int j = 0; j < N; j++) {
                    buff[i / size * N + j] = matrix[i][j];
                }
            }
            int count = p < N% size ? N / size + 1 : N / size;
            MPI_Send(buff, count * N, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
        }
    }
    // 非0号进程负责任务的接收工作
    else {
        MPI_Recv(&matrix[rank][0], task_num * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < task_num; i++) {
            for (int j = 0; j < N; j++) {
                matrix[rank + i * size][j] = matrix[rank + i][j];
            }
        }
    }
    // 做消元运算
    int i, j, k;
#pragma omp parallel num_threads(NUM_THREADS) default(none) private(i, j, k) shared(matrix, N, size, rank)
    for (k = 0; k < N; k++) {
        // 如果除法操作是本进程负责的任务，并将除法结果广播
#pragma omp single
        {
            if (k % size == rank) {
                float32x4_t Akk = vdupq_n_f32(matrix[k][k]);
                for (j = k + 1; j + 3 < N; j += 4) {
                    float32x4_t Akj = vld1q_f32(matrix[k] + j);
                    Akj = vdivq_f32(Akj, Akk);
                    vst1q_f32(matrix[k] + j, Akj);
                }
                // 串行处理结尾
                for (; j < N; j++) {
                    matrix[k][j] = matrix[k][j] / matrix[k][k];
                }
                matrix[k][k] = 1;
                for (int p = 0; p < size; p++) {
                    if (p != rank) {
                        MPI_Send(&matrix[k][0], N, MPI_FLOAT, p, 1, MPI_COMM_WORLD);
                    }
                }
            }
            // 其余进程接收除法行的结果
            else {
                MPI_Recv(&matrix[k][0], N, MPI_FLOAT, k % size, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        // 进行消元操作
        int begin = N / size * size + rank < N ? N / size * size + rank : N / size * size + rank - size;
#pragma omp for schedule(guided)
        for (i = begin; i > k; i -= size) {
            float32x4_t Aik = vdupq_n_f32(matrix[i][k]);
            for (j = k + 1; j + 3 < N; j += 4) {
                float32x4_t Akj = vld1q_f32(matrix[k] + j);
                float32x4_t Aij = vld1q_f32(matrix[i] + j);
                float32x4_t AikMulAkj = vmulq_f32(Aik, Akj);
                Aij = vsubq_f32(Aij, AikMulAkj);
                vst1q_f32(matrix[i] + j, Aij);
            }
            // 串行处理结尾
            for (; j < N; j++) {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
    end_time = MPI_Wtime();
    return (end_time - start_time) * 1000;
}
double calculate_MPI_pipeline(float**matrix,int N) {
    double start_time, end_time;

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        init(matrix,N);
    }
    start_time = MPI_Wtime();
    int task_num = rank < N % size ? N / size + 1 : N / size;

    auto* buff = new float[task_num * N];
    if (rank == 0) {
        for (int p = 1; p < size; p++) {
            for (int i = p; i < N; i += size) {
                for (int j = 0; j < N; j++) {
                    buff[i / size * N + j] = matrix[i][j];
                }
            }
            int count = p < N % size ? N / size + 1 : N / size;
            MPI_Send(buff, count * N, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(&matrix[rank][0], task_num * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < task_num; i++) {
            for (int j = 0; j < N; j++) {
                matrix[rank + i * size][j] = matrix[rank + i][j];
            }
        }
    }

    int pre_proc = (rank + (size - 1)) % size;
    int next_proc = (rank + 1) % size;
    for (int k = 0; k < N; k++) {
        // 如果除法操作是本进程负责的任务，并将除法结果转发给下一个进程
        if (k % size == rank) {
            for (int j = k + 1; j < N; j++) {
                matrix[k][j] /= matrix[k][k];
            }
            matrix[k][k] = 1;
            MPI_Send(&matrix[k][0], N, MPI_FLOAT, next_proc, 1, MPI_COMM_WORLD);
        } else {
            // 其余进程接收前一个进程转发的除法结果，并转发给下一个进程
            MPI_Recv(&matrix[k][0], N, MPI_FLOAT, pre_proc, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (next_proc != k % size) {
                MPI_Send(&matrix[k][0], N, MPI_FLOAT, next_proc, 1, MPI_COMM_WORLD);
            }
        }

        // 进行消去操作
        int begin = N / size * size + rank < N ? N / size * size + rank : N / size * size + rank - size;
        for (int i = begin; i > k; i -= size) {
            for (int j = k + 1; j < N; j++) {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }

    end_time = MPI_Wtime();
    return (end_time - start_time) * 1000;
}



void test(int N) {
    int LOOP = 5;
    int rank;
    float** matrix = new float*[N];
        for (int j = 0; j < N; j++)
        {
            matrix[j] = new float[N];
        }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == 0) {
        cout << "=================================== " << N << " ===================================" << endl;
    }
    struct timeval start {};
    struct timeval end {};
    double time = 0;

    //平凡算法
    time = 0;
    for (int i = 0; i < LOOP; i++) {
        init(matrix,N);
        gettimeofday(&start, nullptr);
        normal(matrix,N);
        gettimeofday(&end, nullptr);
        time += ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) * 1.0 / 1000;
    }
    if (rank == 0) {
        cout << "normal:" << time / LOOP << "ms" << endl;
    }
    //一维行块划分
    time = 0;
    for (int i = 0; i < LOOP; i++) {
        time += calculate_MPI_block(matrix, N);
    }
    if (rank == 0) {
        cout << "MPI_block_row:" << time / LOOP << "ms" << endl;
    }
    //一维行块划分+负载均衡
    time = 0;
    for (int i = 0; i < LOOP; i++) {
        time += calculate_MPI_block_average(matrix, N);
    }
    if (rank == 0) {
        cout << "MPI_block_row_average:" << time / LOOP << "ms" << endl;
    }
  //一维列块划分
   time = 0;
    for (int i = 0; i < LOOP; i++) {
        time += calculate_MPI_col_block(matrix, N);
    }
    if (rank == 0) {
        cout << "MPI_block_col:" << time / LOOP << "ms" << endl;
    }
    //循环划分
    time = 0;
    for (int i = 0; i < LOOP; i++) {
        time += calculate_MPI_SIMD(matrix, N);
    }
    if (rank == 0) {
        cout << "MPI_SIMD:" << time / LOOP << "ms" << endl;
    }
    time = 0;
    for (int i = 0; i < LOOP; i++) {
        time += calculate_MPI_cycle(matrix, N);
    }
    if (rank == 0) {
        cout << "MPI_CYCLE:" << time / LOOP << "ms" << endl;
    }
    time = 0;
    for (int i = 0; i < LOOP; i++) {
        time +=  calculate_MPI_pipeline(matrix, N);
    }
    if (rank == 0) {
        cout << "MPI_Pipeline:" << time / LOOP << "ms" << endl;
    }
   
    time = 0;
    for (int i = 0; i < LOOP; i++) {
        time += calculate_MPI_OMP(matrix, N);
    }
    if (rank == 0) {
        cout << "MPI_OpenMP:" << time / LOOP << "ms" << endl;
    }
    time = 0;
    for (int i = 0; i < LOOP; i++) {
        time += calculate_MPI_OMP_SIMD(matrix, N);
    }
    if (rank == 0) {
        cout << "MPI_SIMD_OpenMP:" << time / LOOP << "ms" << endl;
    }
    
}

int main()
{
    MPI_Init(nullptr, nullptr);
    for (int i = 500; i <= 4000; i += 500)
    {
        test(i);
    }
    MPI_Finalize();
    return 0;
}