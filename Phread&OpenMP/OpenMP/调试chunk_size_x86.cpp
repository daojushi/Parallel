#include <pthread.h>
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>
#include <semaphore.h>
#include <unistd.h>
#include <omp.h>
using namespace std;
#define NUM_THREADS1 8//以8个线程，dynamic为例调试CHUNK_SIZE
#define CHUNK_SIZE1 8
#define CHUNK_SIZE2 6
#define CHUNK_SIZE3 4
void init(float** a,int n) {//测试用例生成
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            a[i][j] = 0;
        }
        a[i][i] = 1.0;
        for(int j = i + 1; j < n; j++) {
            a[i][j] = rand();
        }
    }
    for(int k = 0; k < n; k++) {
        for(int i = k + 1; i < n; i++) {
            for(int j = 0; j < n; j++) {
                a[i][j] += a[k][j];
            }
        }
    }
}
void OpenMP_Dynamic8(float** a, int n)////chunk_size为8
{
    #pragma omp parallel for schedule(dynamic, CHUNK_SIZE1) num_threads(NUM_THREADS1) shared(a,n)
    for (int k = 0; k < n; k++)
    {
        for (int j = k + 1; j < n; j++)
            a[k][j] = a[k][j] / a[k][k];
        a[k][k] = 1.0;
        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j < n; j++)
                a[i][j] = a[i][j] - a[i][k] * a[k][j];
            a[i][k] = 0;
        }
    }
}
void OpenMP_Dynamic6(float** a, int n)//chunk_size为6
{
    #pragma omp parallel for schedule(dynamic, CHUNK_SIZE2) num_threads(NUM_THREADS1) shared(a,n)
    for (int k = 0; k < n; k++)
    {
        for (int j = k + 1; j < n; j++)
            a[k][j] = a[k][j] / a[k][k];
        a[k][k] = 1.0;
        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j < n; j++)
                a[i][j] = a[i][j] - a[i][k] * a[k][j];
            a[i][k] = 0;
        }
    }
}
void OpenMP_Dynamic4(float** a, int n)//chunk_size为4
{
    #pragma omp parallel for schedule(dynamic, CHUNK_SIZE3) num_threads(NUM_THREADS1) shared(a,n)
    for (int k = 0; k < n; k++)
    {
        for (int j = k + 1; j < n; j++)
            a[k][j] = a[k][j] / a[k][k];
        a[k][k] = 1.0;
        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j < n; j++)
                a[i][j] = a[i][j] - a[i][k] * a[k][j];
            a[i][k] = 0;
        }
    }
}
void OpenMP_Dynamic(float** a, int n)//采用默认值
{
    #pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS1) shared(a,n)
    for (int k = 0; k < n; k++)
    {
        for (int j = k + 1; j < n; j++)
            a[k][j] = a[k][j] / a[k][k];
        a[k][k] = 1.0;
        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j < n; j++)
                a[i][j] = a[i][j] - a[i][k] * a[k][j];
            a[i][k] = 0;
        }
    }
}
int main()
{
    int n=1500;
    float** test = new float*[n];
        for (int j = 0; j < n; j++)
        {
            test[j] = new float[n];
        }
        init(test,n);
        srand(time(NULL));
        LARGE_INTEGER timeStart;	//开始时间
        LARGE_INTEGER timeEnd;		//结束时间
        LARGE_INTEGER frequency;	//计时器频率
        QueryPerformanceFrequency(&frequency);
        double quadpart = (double)frequency.QuadPart;//计时器频率

        QueryPerformanceCounter(&timeStart);
        OpenMP_Dynamic8(test,n);
        QueryPerformanceCounter(&timeEnd);
        double _d8 = (timeEnd.QuadPart - timeStart.QuadPart)*1000 / quadpart;
        cout<<"OpenMP_Dynamic8:"<<_d8<<"ms"<<endl;
        init(test,n);


        QueryPerformanceCounter(&timeStart);
        OpenMP_Dynamic6(test,n);
        QueryPerformanceCounter(&timeEnd);
        double _d6 = (timeEnd.QuadPart - timeStart.QuadPart)*1000 / quadpart;
        cout<<"OpenMP_Dynamic6:"<<_d6<<"ms"<<endl;
        init(test,n);


        QueryPerformanceCounter(&timeStart);
        OpenMP_Dynamic4(test,n);
        QueryPerformanceCounter(&timeEnd);
        double _d4 = (timeEnd.QuadPart - timeStart.QuadPart)*1000 / quadpart;
        cout<<"OpenMP_Dynamic4:"<<_d4<<"ms"<<endl;

        QueryPerformanceCounter(&timeStart);
        OpenMP_Dynamic4(test,n);
        QueryPerformanceCounter(&timeEnd);
        double _d = (timeEnd.QuadPart - timeStart.QuadPart)*1000 / quadpart;
        cout<<"OpenMP_Dynamic:"<<_d<<"ms"<<endl;
}
