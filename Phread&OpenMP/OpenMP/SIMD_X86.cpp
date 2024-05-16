#include <pthread.h>
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>
#include <semaphore.h>
#include <unistd.h>
#include <omp.h>
#include <immintrin.h>//for AVX2
#include <emmintrin.h>// for SSE2
using namespace std;
#define NUM_THREADS 7
//以dynamic为例
void sse(float** a, int n)
{
    #pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS) shared(a,n)
    for (int k = 0; k < n; k++)
    {
        __m128 vk = _mm_set1_ps(a[k][k]);
        a[k][k] = 1.0;
        for (int j = k + 1; j < n; j += 4)
        {
            __m128 vaj = _mm_loadu_ps(&a[k][j]);
            __m128 vresult = _mm_div_ps(vaj, vk);
            _mm_storeu_ps(&a[k][j], vresult);
        }
        for (int i = k + 1; i < n; i++)
        {
            __m128 vik = _mm_set1_ps(a[i][k]);
            for (int j = k + 1; j < n; j += 4)
            {
                __m128 vaj = _mm_loadu_ps(&a[k][j]);
                __m128 vij = _mm_loadu_ps(&a[i][j]);
                __m128 vmul = _mm_mul_ps(vik, vaj);
                __m128 vresult = _mm_sub_ps(vij, vmul);
                _mm_storeu_ps(&a[i][j], vresult);
            }
            a[i][k] = 0;
        }
    }
}
void avx(float** a, int n)
{
    #pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS) shared(a,n)
    for (int k = 0; k < n; k++)
    {
        __m256 vk = _mm256_set1_ps(a[k][k]);
        a[k][k] = 1.0;
        for (int j = k + 1; j < n; j += 8)
        {
            __m256 vaj = _mm256_loadu_ps(&a[k][j]);
            __m256 vresult = _mm256_div_ps(vaj, vk);
            _mm256_storeu_ps(&a[k][j], vresult);
        }
        for (int i = k + 1; i < n; i++)
        {
            __m256 vik = _mm256_set1_ps(a[i][k]);
            for (int j = k + 1; j < n; j += 8)
            {
                __m256 vaj = _mm256_loadu_ps(&a[k][j]);
                __m256 vij = _mm256_loadu_ps(&a[i][j]);
                __m256 vmul = _mm256_mul_ps(vik, vaj);
                __m256 vresult = _mm256_sub_ps(vij, vmul);
                _mm256_storeu_ps(&a[i][j], vresult);
            }
            a[i][k] = 0;
        }
    }
}
void Dynamic(float** a, int n)
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
void SIMD_auto(float** a, int n)
{
    #pragma omp parallel for simd schedule(dynamic) num_threads(NUM_THREADS) shared(a,n)
    for (int k = 0; k < n; k++)
    {
        for (int j = k + 1; j < n; j ++)
        {
            a[k][j] /= a[k][k];
        }
        a[k][k] = 1.0;
        for (int i = k + 1; i < n; i++)
        {
            for (int j = k + 1; j < n; j++)
            {
                a[i][j] -= a[i][k] * a[k][j];
            }
            a[i][k] = 0.0;
        }
    }
}
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


        //不使用SIMD
        QueryPerformanceCounter(&timeStart);
        Dynamic(test,n[i]);
        QueryPerformanceCounter(&timeEnd);
        double _s8 = (timeEnd.QuadPart - timeStart.QuadPart)*1000 / quadpart;
        cout<<"No SIMD:"<<_s8<<"ms"<<endl;
        init(test,n[i]);

        //使用sse
        QueryPerformanceCounter(&timeStart);
        sse(test,n[i]);
        QueryPerformanceCounter(&timeEnd);
        double _s6 = (timeEnd.QuadPart - timeStart.QuadPart)*1000 / quadpart;
        cout<<"openmp+sse:"<<_s6<<"ms"<<endl;
        init(test,n[i]);

        //使用avx
        QueryPerformanceCounter(&timeStart);
        avx(test,n[i]);
        QueryPerformanceCounter(&timeEnd);
        double _s4 = (timeEnd.QuadPart - timeStart.QuadPart)*1000 / quadpart;
        cout<<"openmp+avx:"<<_s4<<"ms"<<endl;
        init(test,n[i]);

        //自动向量化
        QueryPerformanceCounter(&timeStart);
        SIMD_auto(test,n[i]);
        QueryPerformanceCounter(&timeEnd);
        double _d8 = (timeEnd.QuadPart - timeStart.QuadPart)*1000 / quadpart;
        cout<<"SIMD_auto:"<<_d8<<"ms"<<endl;


    }
    return 0;
}
