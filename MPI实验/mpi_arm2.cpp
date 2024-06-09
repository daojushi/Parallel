#include <iostream>
#include <mpi.h>
#include <fstream>
#include <cmath>
#include <sys/time.h>
#include <arm_neon.h>
#include <omp.h>
#include<cstring>
using namespace std;



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
//һά�п黮��

double calculate_MPI_block(float** matrix,int N) {
    double start_time, end_time;

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // ֻ����0�Ž��̣��Ž��г�ʼ������
    if (rank == 0) {
        init(matrix,N);
    }
    start_time = MPI_Wtime();
    int task_num = ceil(N * 1.0 / size);
    // 0�Ž��̸�������ĳ�ʼ�ַ�����
    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            int start = i * task_num;
            int end = (i + 1) * task_num;
            if (i == size - 1){
                end = N;
            }
            MPI_Send(&matrix[start][0], (end - start) * N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
        }
    }
        // ��0�Ž��̸�������Ľ��չ���
    else {
        if (rank != size - 1) {
            MPI_Recv(&matrix[rank * task_num][0], task_num * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(&matrix[rank * task_num][0], (N - rank * task_num) * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }
    }

    // ����Ԫ����
    int start = rank * task_num;
    int end = (rank + 1) * task_num < N ? (rank + 1) * task_num : N;
    for (int k = 0; k < N; k++) {
        // ������������Ǳ����̸�������񣬲�����������㲥
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
            // ������̽��ճ����еĽ��
        else {
            MPI_Recv(&matrix[k][0], N, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // ������Ԫ����
        for (int i = max(k + 1, start); i < end; i++) {
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

    //һά�п黮��
    float time = 0;
    for (int i = 0; i < LOOP; i++) {
        time += calculate_MPI_block(matrix, N);
    }
    if (rank == 0) {
        cout << "MPI_block_row:" << time / LOOP << "ms" << endl;
    }

}

int main()
{
    MPI_Init(nullptr, nullptr);
    for (int i = 500; i <= 500; i += 500)
    {
        test(i);
    }
    MPI_Finalize();
    return 0;
}
