#include <pthread.h>
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>
#include <semaphore.h>
#include <unistd.h>
#include <omp.h>
using namespace std;
#define NUM_THREADS 8
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
void OpenMP_pad(float** a, int n)
{
    const int CACHE_LINE_SIZE = 64;
    int padding_size = CACHE_LINE_SIZE / sizeof(float);
    #pragma omp parallel for num_threads(NUM_THREADS) shared(a, n) schedule(dynamic)
    for (int k = 0; k < n; k++)
    {
        for (int j = k + 1; j < n; j ++)
        {
            a[k][j] = a[k][j] / a[k][k];
        }
        a[k][k] = 1.0;
        for (int i = k + 1; i < n; i++)
        {
            float padding[padding_size];
            for (int p = 0; p < padding_size; p++) {
                padding[p] = 0.0;
            }
            for (int j = k + 1; j < n; j++)
            {
                a[i][j] -= a[i][k] * a[k][j];
            }
            a[i][k] = 0;
        }
    }
}
void OpenMP_Static(float** a, int n)//使用静态调度，6个线程
{
    #pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS) shared(a,n)
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
        int n[8]={64,128,256,512,1024,1536,2048,3000};
    for(int i=0;i<8;i++)
    {
        float** test = new float*[n[i]];
        for (int j = 0; j < n[i]; j++)
        {
            test[j] = new float[n[i]];
        }
        init(test,n[i]);
        cout<<"N is:"<<n[i]<<endl;
        srand(time(NULL));
        LARGE_INTEGER timeStart;	//开始时间
        LARGE_INTEGER timeEnd;		//结束时间
        LARGE_INTEGER frequency;	//计时器频率
        QueryPerformanceFrequency(&frequency);
        double quadpart = (double)frequency.QuadPart;//计时器频率
        QueryPerformanceCounter(&timeStart);
        OpenMP_Static(test,n[i]);
        QueryPerformanceCounter(&timeEnd);
        double _d8 = (timeEnd.QuadPart - timeStart.QuadPart)*1000 / quadpart;
        cout<<"OpenMP_Static:"<<_d8<<"ms"<<endl;
        init(test,n[i]);


        QueryPerformanceCounter(&timeStart);
        OpenMP_pad(test,n[i]);
        QueryPerformanceCounter(&timeEnd);
        double _d6 = (timeEnd.QuadPart - timeStart.QuadPart)*1000 / quadpart;
        cout<<"OpenMP_pad:"<<_d6<<"ms"<<endl;
        init(test,n[i]);

    }
    return 0;



}
