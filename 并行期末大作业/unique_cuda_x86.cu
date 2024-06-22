#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cuda_runtime.h>
#include <Windows.h>

const int N = 130; // Matrix size, modify based on your test case
const int BLOCK_SIZE = 256; // CUDA block size

char subMatrix[N][N + 1]; // Elimination submatrix
char eliminatedRows[N][N]; // Rows to be eliminated
int bLength; // Length of rows to be eliminated

__global__ void gaussianElimination(char* subMatrix, char* eliminatedRows, int bLength)
{
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx < bLength) {
        for (int col_idx = N - 1; col_idx >= 0; col_idx--) {
            if (eliminatedRows[row_idx * N + col_idx] == 1) {
                if (subMatrix[col_idx * (N + 1) + N] != 0) {
                    // Sub row exists
                    for (int k = 0; k <= col_idx; k++) {
                        eliminatedRows[row_idx * N + k] = abs(eliminatedRows[row_idx * N + k] - subMatrix[col_idx * (N + 1) + k]);
                    }
                }
                else {
                    // No sub row elimination boosting
                    for (int k = 0; k <= col_idx; k++) {
                        subMatrix[col_idx * (N + 1) + k] = eliminatedRows[row_idx * N + k];
                    }
                    subMatrix[col_idx * (N + 1) + N] = 1; // Mark this row as having data after boosting
                    break;
                }
            }
        }
    }
}

void createSubMatrix(char* fname)
{
    std::ifstream file(fname);
    std::string line;
    int row = 0;

    while (std::getline(file, line) && row < N) {
        std::istringstream iss(line);
        char val;
        int col = 0;

        while (iss >> val && col <= N) {
            subMatrix[row][col] = val;
            col++;
        }

        row++;
    }
}

void createEliminatedRows(char* fname)
{
    std::ifstream file(fname);
    std::string line;
    int row = 0;

    while (std::getline(file, line) && row < N) {
        std::istringstream iss(line);
        char val;
        int col = 0;

        while (iss >> val && col < N) {
            eliminatedRows[row][col] = val;
            col++;
        }

        row++;
    }

    bLength = row;
}

void writeResult(char* fname)
{
    std::ofstream file(fname);

    for (int row = 0; row < N; row++) {
        for (int col = 0; col <= N; col++) {
            file << static_cast<int>(subMatrix[row][col]) << " ";
        }
        file << std::endl;
    }

    file.close();
}

double getElapsedTime(LARGE_INTEGER start, LARGE_INTEGER end, LARGE_INTEGER frequency)
{
    return static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
}

int main()
{
    char* d_subMatrix;
    char* d_eliminatedRows;

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_subMatrix, sizeof(char) * N * (N + 1));
    cudaMalloc((void**)&d_eliminatedRows, sizeof(char) * N * N);

    // Read input files and populate subMatrix and eliminatedRows
    createSubMatrix("1.txt");
    createEliminatedRows("2.txt");

    // Copy data from host to device
    cudaMemcpy(d_subMatrix, subMatrix, sizeof(char) * N * (N + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_eliminatedRows, eliminatedRows, sizeof(char) * N * N, cudaMemcpyHostToDevice);

    // Measure execution time for CUDA implementation
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);

    LARGE_INTEGER start, end;
    QueryPerformanceCounter(&start);

    // Launch CUDA kernel
    int numBlocks = (bLength + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gaussianElimination << <numBlocks, BLOCK_SIZE >> > (d_subMatrix, d_eliminatedRows, bLength);

    // Copy results from device to host
    cudaMemcpy(subMatrix, d_subMatrix, sizeof(char) * N * (N + 1), cudaMemcpyDeviceToHost);

    // Measure end time and calculate execution time
    QueryPerformanceCounter(&end);
    double elapsedSeconds = getElapsedTime(start, end, frequency) * 1000.0;

    // Write results to file
    writeResult("CUDA_GaussianElimination_Result.txt");

    // Cleanup
    cudaFree(d_subMatrix);
    cudaFree(d_eliminatedRows);

    std::cout << "CUDA Execution Time: " << elapsedSeconds << " ms" << std::endl;

    return 0;
}
