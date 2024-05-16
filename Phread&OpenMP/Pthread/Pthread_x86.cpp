#include <iostream>
#include<sys/time.h>
#include<pthread.h>
#include<semaphore.h>
#include<xmmintrin.h>
#include<immintrin.h>
#include<x86intrin.h>
using namespace std;

const int n=100;
float a[n][n];
const int NUM_THREADS=7;

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
void normal(float** a,int n) {
	for (int k = 0; k < n; k++) {
		for (int j = k + 1; j < n; j++)
			a[k][j] /= a[k][k];
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++) {
			for (int j = k + 1; j < n; j++)
				a[i][j] = a[i][j] - a[i][k] * a[k][j];
			a[i][k] = 0;
		}
	}
}

typedef struct{
	int k;
	int t_id;
}threadParam_t;

//动态线程
void* threadFunc(void* param) {
	threadParam_t *p = (threadParam_t*)param;
	int k = p->k;
	int t_id = p->t_id;
	for(int i = k+1+t_id;i<n;i+=NUM_THREADS)
	{
		for(int j=k+1;j<n;++j)
			a[i][j] = a[i][j] - a[i][k]*a[k][j];
		a[i][k] = 0;
	}
	pthread_exit(NULL);
}
void gauss_dynamic() {
    for(int k=0;k<n;++k)
	{
		for(int j=k+1;j<n;j++)
			a[k][j] = a[k][j]/a[k][k];
		a[k][k] = 1.0;

		pthread_t* handles = (pthread_t*)malloc(NUM_THREADS*sizeof(pthread_t));
		threadParam_t* param = (threadParam_t*)malloc(NUM_THREADS*sizeof(param));

		for (int t_id = 0;t_id < NUM_THREADS;t_id++)
		{
			param[t_id].k = k;
			param[t_id].t_id = t_id;
		}
		for (int t_id = 0;t_id<NUM_THREADS;t_id++)
			pthread_create(&handles[t_id],NULL,threadFunc,(void*)&param[t_id]);
		for (int t_id = 0;t_id<NUM_THREADS;t_id++)
			pthread_join(handles[t_id],NULL);
    }
}

//静态线程+信号量同步

sem_t sem_main;
sem_t sem_workerstart[NUM_THREADS];
sem_t sem_workerend[NUM_THREADS];
void* threadFunc1(void *param)
{
    threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;

	for(int k = 0;k<n;++k)
	{
		sem_wait(&sem_workerstart[t_id]);
		for(int i = k+1+t_id;i<n;i+=NUM_THREADS)
		{
			for(int j=k+1;j<n;++j)
				a[i][j] = a[i][j] - a[i][j] * a[k][j];
			a[i][k]=0.0;
		}
		sem_post(&sem_main);
		sem_wait(&sem_workerend[t_id]);
	}
	pthread_exit(NULL);
}
void gauss_static1()
{
    sem_init(&sem_main, 0, 0);
    for (int i = 0; i < NUM_THREADS; ++i)
    {
        sem_init(&sem_workerstart[i], 0, 0);
        sem_init(&sem_workerend[i], 0, 0);
    }
    pthread_t* handles = new pthread_t[NUM_THREADS];
    threadParam_t* param = new threadParam_t[NUM_THREADS];
    for (int t_id = 0;t_id < NUM_THREADS;t_id++)
    {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id],NULL,threadFunc1,(void*)&param[t_id]);
    }

    for (int k = 0;k < n;++k)
    {
        for (int j = k + 1;j < n;j++)a[k][j] = a[k][j] / a[k][k];

        a[k][k] = 1.0;

        for (int t_id = 0;t_id < NUM_THREADS;++t_id)sem_post(&sem_workerstart[t_id]);
        for (int t_id = 0;t_id < NUM_THREADS;++t_id)sem_wait(&sem_main);
        for (int t_id = 0;t_id < NUM_THREADS;++t_id)sem_post(&sem_workerend[t_id]);

    }
    for (int t_id = 0;t_id < NUM_THREADS;t_id++)pthread_join(handles[t_id],NULL);
    sem_destroy(&sem_main);
    for (int t_id = 0;t_id < NUM_THREADS;t_id++)sem_destroy(&sem_workerstart[t_id]);
    for (int t_id = 0;t_id < NUM_THREADS;t_id++)sem_destroy(&sem_workerend[t_id]);
}

//静态线程+信号量同步+三重循环全部纳入线程函数
sem_t sem_Divsion;
sem_t sem_Elimination;
void* threadFunc2(void* param)
{
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;

    for (int k = 0;k < n;++k)
    {
        if(t_id==0)
        {
            for (int j = k + 1;j < n;++j)a[k][j] = a[k][j] / a[k][k];
            a[k][k] = 1.0;
        }
        else {
            sem_wait(&sem_Divsion);
        }
        if(t_id==0)
        {
            for(int t_id=0;t_id<NUM_THREADS-1;++t_id)sem_post(&sem_Divsion);
        }
        for (int i = k + 1 + t_id;i < n;i += NUM_THREADS)
        {
            for (int j = k + 1;j < n;++j)a[i][j] = a[i][j] - a[i][j] * a[k][j];
            a[i][k] = 0.0;
        }
        if (t_id == 0)
        {
            for (int t_id = 0;t_id < NUM_THREADS - 1;++t_id)sem_post(&sem_Elimination);
        }
        else {
            sem_wait(&sem_Elimination);
        }
    }
    pthread_exit(NULL);
}
void gauss_static2()
{
    sem_init(&sem_Divsion, 0, 0);
    sem_init(&sem_Elimination, 0, 0);

    pthread_t handles[NUM_THREADS];
    threadParam_t param[NUM_THREADS];
    for (int t_id = 0;t_id < NUM_THREADS;t_id++)
    {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id],NULL,threadFunc2,(void*)&param[t_id]);
    }
    for (int t_id = 0;t_id < NUM_THREADS;t_id++)pthread_join(handles[t_id],NULL);

    sem_destroy(&sem_Divsion);
    sem_destroy(&sem_Elimination);
}

//静态线程 +barrier 同步
pthread_barrier_t barrier_Divsion;
pthread_barrier_t barrier_Elimination;

void* threadFunc3(void* param)
{
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    for (int k = 0;k < n;++k)
    {
        if(t_id==0)
        {
            for (int j = k + 1;j < n;++j)a[k][j] = a[k][j] / a[k][k];
            a[k][k] = 1.0;
        }  pthread_barrier_wait(&barrier_Divsion);
        for (int i = k + 1 + t_id;i < n;i += NUM_THREADS)
        {
            for (int j = k + 1;j < n;++j)a[i][j] = a[i][j] - a[i][j] * a[k][j];
            a[i][k] = 0.0;
        }
        pthread_barrier_wait(&barrier_Elimination);
    }
    pthread_exit(NULL);
}
void gauss_static3()
{
    pthread_barrier_init(&barrier_Divsion, NULL, NUM_THREADS);
    pthread_barrier_init(&barrier_Elimination, NULL, NUM_THREADS);

    pthread_t handles[NUM_THREADS];
    threadParam_t param[NUM_THREADS];
    for (int t_id = 0;t_id < NUM_THREADS;t_id++)
    {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id],NULL,threadFunc3,(void*)&param[t_id]);
    }
    for (int t_id = 0;t_id < NUM_THREADS;t_id++)pthread_join(handles[t_id],NULL);

    pthread_barrier_destroy(&barrier_Divsion);
    pthread_barrier_destroy(&barrier_Elimination);
}

//静态线程行分配

void* threadFunc4(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    for(int k = 0; k < n; k++) {
        sem_wait(&sem_workerstart[t_id]);
        for(int i = k + t_id + 1; i < n; i += NUM_THREADS) {
            for(int j = k + 1; j < n; j++) {
                a[i][j] = a[i][j] - a[i][k]*a[k][j];
            }
            a[i][k] = 0.0;
        }
        sem_post(&sem_main);
        sem_wait(&sem_workerend[t_id]);
    }
    pthread_exit(NULL);
}
void gauss_static_row() {
        sem_init(&sem_main, 0, 0);
        for(int i = 0; i < NUM_THREADS; i++) {
            sem_init(&sem_workerstart[i], 0, 0);
            sem_init(&sem_workerend[i], 0, 0);
        }
        pthread_t handles[NUM_THREADS];
        threadParam_t param[NUM_THREADS];
        for(int t_id = 0; t_id < NUM_THREADS; t_id++) {
            param[t_id].t_id = t_id;
            pthread_create(handles + t_id, NULL, threadFunc4, param + t_id);
        }
        for(int k = 0; k < n; k++) {
            for(int j = k + 1; j < n; j++) {
                a[k][j] = a[k][j]/a[k][k];
            }
            a[k][k] = 1.0;
            for(int t_id = 0; t_id <NUM_THREADS; t_id++) {
                sem_post(&sem_workerstart[t_id]);
            }
            for(int t_id = 0; t_id < NUM_THREADS; t_id++) {
                sem_wait(&sem_main);
            }
            for(int t_id = 0; t_id < NUM_THREADS; t_id++) {
                sem_post(&sem_workerend[t_id]);
            }
        }
        for(int t_id = 0; t_id < NUM_THREADS; t_id++) {
            pthread_join(handles[t_id], NULL);
        }
        sem_destroy(&sem_main);
        for(int i = 0; i < NUM_THREADS; i++) {
            sem_destroy(&sem_workerstart[i]);
            sem_destroy(&sem_workerend[i]);
        }
}
//静态线程列分配
void* threadFunc5(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    for(int k = 0; k < n; k++) {
        sem_wait(&sem_workerstart[t_id]);
        for(int i = k + 1; i < n; i++) {
            for(int j = k + t_id + 1; j < n; j +=NUM_THREADS) {
                a[i][j] = a[i][j] - a[i][k]*a[k][j];
            }
            a[i][k] = 0.0;
        }
        sem_post(&sem_main);
        sem_wait(&sem_workerend[t_id]);
    }
     pthread_exit(NULL);
}
void gauss_static_col() {
        sem_init(&sem_main, 0, 0);
        for(int i = 0; i < NUM_THREADS; i++) {
            sem_init(&sem_workerstart[i], 0, 0);
            sem_init(&sem_workerend[i], 0, 0);
        }
        pthread_t handles[NUM_THREADS];
        threadParam_t param[NUM_THREADS];
        for(int t_id = 0; t_id < NUM_THREADS; t_id++) {
            param[t_id].t_id = t_id;
            pthread_create(handles + t_id, NULL, threadFunc5, param + t_id);
        }
        for(int k = 0; k <NUM_THREADS; k++) {
            for(int j = k + 1; j <NUM_THREADS; j++) {
                a[k][j] = a[k][j]/a[k][k];
            }
            a[k][k] = 1.0;
            for(int t_id = 0; t_id <NUM_THREADS; t_id++) {
                sem_post(&sem_workerstart[t_id]);
            }
            for(int t_id = 0; t_id < NUM_THREADS; t_id++) {
                sem_wait(&sem_main);
            }
            for(int t_id = 0; t_id <NUM_THREADS; t_id++) {
                sem_post(&sem_workerend[t_id]);
            }
        }
        for(int t_id = 0; t_id <NUM_THREADS; t_id++) {
            pthread_join(handles[t_id], NULL);
        }
        sem_destroy(&sem_main);
        for(int i = 0; i <NUM_THREADS; i++) {
            sem_destroy(&sem_workerstart[i]);
            sem_destroy(&sem_workerend[i]);
        }
}

//静态线程行分配+SSE
void* threadFunc6(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    for(int k = 0; k < n; k++) {
        sem_wait(&sem_workerstart[t_id]);
        for(int i = k + t_id + 1; i < n; i += NUM_THREADS) {
            float* aik = new float[4];
            *aik = *(aik+1) = *(aik+2) = *(aik+3) = a[i][k];
            __m128 vaik = _mm_load_ps(aik);
            delete[] aik;
            for(int j = k + 1; j + 3 < n; j += 4) {
                __m128 vaij = _mm_loadu_ps(&a[i][j]);
                __m128 vakj = _mm_loadu_ps(&a[k][j]);
                __m128 vx = _mm_mul_ps(vaik, vakj);
                vaij = _mm_sub_ps(vaij, vx);
                _mm_storeu_ps(&a[i][j], vaij);
            }
            for(int j = n - n%4; j < n; j++) {
                a[i][j] = a[i][j] - a[i][k]*a[k][j];
            }
            a[i][k] = 0;
        }
        sem_post(&sem_main);
        sem_wait(&sem_workerend[t_id]);
    }
    pthread_exit(NULL);
}
void gauss_static_row_sse() {
        sem_init(&sem_main, 0, 0);
        for(int i = 0; i < NUM_THREADS; i++) {
            sem_init(&sem_workerstart[i], 0, 0);
            sem_init(&sem_workerend[i], 0, 0);
        }
        pthread_t handles[NUM_THREADS];
        threadParam_t param[NUM_THREADS];
        for(int t_id = 0; t_id < NUM_THREADS; t_id++) {
            param[t_id].t_id = t_id;
            pthread_create(handles + t_id, NULL, threadFunc6, param + t_id);
        }
        for(int k = 0; k < n; k++) {
            float* akk = new float[4];
            *akk = *(akk+1) = *(akk+2) = *(akk+3) = a[k][k];
            __m128 vt = _mm_load_ps(akk);
            delete[] akk;
            for(int j = k + 1; j + 3 < n; j += 4) {
                __m128 va = _mm_loadu_ps(&a[k][j]);
                va = _mm_div_ps(va, vt);
                _mm_storeu_ps(&a[k][j], va);
            }
            for(int j = n - n%4; j < n; j++) {
                a[k][j] = a[k][j]/a[k][k];
            }
            a[k][k] = 1.0;
            for(int t_id = 0; t_id < NUM_THREADS; t_id++) {
                sem_post(&sem_workerstart[t_id]);
            }
            for(int t_id = 0; t_id < NUM_THREADS; t_id++) {
                sem_wait(&sem_main);
            }
            for(int t_id = 0; t_id < NUM_THREADS; t_id++) {
                sem_post(&sem_workerend[t_id]);
            }
        }
        for(int t_id = 0; t_id < NUM_THREADS; t_id++) {
            pthread_join(handles[t_id], NULL);
        }
        sem_destroy(&sem_main);
        for(int i = 0; i < NUM_THREADS; i++) {
            sem_destroy(&sem_workerstart[i]);
            sem_destroy(&sem_workerend[i]);
        }
}

//静态线程行分配+AVX
void* threadFunc7(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    for(int k = 0; k < n; k++) {
        sem_wait(&sem_workerstart[t_id]);
        for(int i = k + t_id + 1; i < n; i += NUM_THREADS) {
            float* aik = new float[8];
            *aik = *(aik+1) = *(aik+2) = *(aik+3) = *(aik+4) = *(aik+5) = *(aik+6) = *(aik+7) = a[i][k];
            __m256 vaik = __builtin_ia32_loadups256(aik);
            delete[] aik;
            for(int j = k + 1; j + 7 < n; j += 8) {
                __m256 vaij = __builtin_ia32_loadups256(&a[i][j]);
                __m256 vakj = __builtin_ia32_loadups256(&a[k][j]);
                __m256 vx = __builtin_ia32_mulps256(vaik, vakj);
                vaij = __builtin_ia32_subps256(vaij, vx);
                __builtin_ia32_storeups256(&a[i][j], vaij);
            }
            for(int j = n - n%8; j < n; j++) {
                a[i][j] = a[i][j] - a[i][k]*a[k][j];
            }
            a[i][k] = 0;
        }
        sem_post(&sem_main);
        sem_wait(&sem_workerend[t_id]);
    }
    pthread_exit(NULL);
}
void gauss_static_row_avx() {
        sem_init(&sem_main, 0, 0);
        for(int i = 0; i < NUM_THREADS; i++) {
            sem_init(&sem_workerstart[i], 0, 0);
            sem_init(&sem_workerend[i], 0, 0);
        }
        pthread_t handles[NUM_THREADS];
        threadParam_t param[NUM_THREADS];
        for(int t_id = 0; t_id < NUM_THREADS; t_id++) {
            param[t_id].t_id = t_id;
            pthread_create(handles + t_id, NULL, threadFunc4, param + t_id);
        }
        for(int k = 0; k < n; k++) {
            float* akk = new float[8];
            *akk = *(akk+1) = *(akk+2) = *(akk+3) = *(akk+4) = *(akk+5) = *(akk+6) = *(akk+7) = a[k][k];
            __m256 vt = __builtin_ia32_loadups256(akk);
            delete[] akk;
            for(int j = k + 1; j + 7 < n; j += 8) {
                __m256 va = __builtin_ia32_loadups256(&a[k][j]);
                va = _mm256_div_ps(va, vt);
                __builtin_ia32_storeups256(&a[k][j], va);
            }
            for(int j = n - n%8; j < n; j++) {
                a[k][j] = a[k][j]/a[k][k];
            }
            a[k][k] = 1.0;
            for(int t_id = 0; t_id < NUM_THREADS; t_id++) {
                sem_post(&sem_workerstart[t_id]);
            }
            for(int t_id = 0; t_id < NUM_THREADS; t_id++) {
                sem_wait(&sem_main);
            }
            for(int t_id = 0; t_id < NUM_THREADS; t_id++) {
                sem_post(&sem_workerend[t_id]);
            }
        }
        for(int t_id = 0; t_id < NUM_THREADS; t_id++) {
            pthread_join(handles[t_id], NULL);
        }
        sem_destroy(&sem_main);
        for(int i = 0; i < NUM_THREADS; i++) {
            sem_destroy(&sem_workerstart[i]);
            sem_destroy(&sem_workerend[i]);
        }
}
int main()
{

        float** a = new float*[n];
        for (int j = 0; j < n; j++)
        {
        a[j] = new float[n];
        }
        cout<<"本次实验规模为"<<n<<endl;
        init(a,n);
       srand(time(NULL));
        LARGE_INTEGER timeStart;	//开始时间
        LARGE_INTEGER timeEnd;		//结束时间
        LARGE_INTEGER frequency;	//计时器频率
        QueryPerformanceFrequency(&frequency);
        double quadpart = (double)frequency.QuadPart;//计时器频率

        //平凡算法
        QueryPerformanceCounter(&timeStart);
        normal(a,n);
        QueryPerformanceCounter(&timeEnd);
        double _Trivial = (timeEnd.QuadPart - timeStart.QuadPart)*1000 / quadpart;
        cout<<"平凡算法:"<<_Trivial<<"ms"<<endl;
        init(a,n);

       //动态线程
        QueryPerformanceCounter(&timeStart);
        gauss_dynamic();
        QueryPerformanceCounter(&timeEnd);
        _Trivial = (timeEnd.QuadPart - timeStart.QuadPart)*1000 / quadpart;
        cout<<"动态线程算法:"<<_Trivial<<"ms"<<endl;
        init(a,n);
       //静态线程+信号量同步
        QueryPerformanceCounter(&timeStart);
        gauss_static1();
        QueryPerformanceCounter(&timeEnd);
        _Trivial = (timeEnd.QuadPart - timeStart.QuadPart)*1000 / quadpart;
        cout<<"静态线程+信号量同步:"<<_Trivial<<"ms"<<endl;
        init(a,n);
       //静态线程+信号量同步+三重循环全部纳入线程函数
        QueryPerformanceCounter(&timeStart);
        gauss_static2();
        QueryPerformanceCounter(&timeEnd);
        _Trivial = (timeEnd.QuadPart - timeStart.QuadPart)*1000 / quadpart;
        cout<<"静态线程+信号量同步+三重循环全部纳入线程函数:"<<_Trivial<<"ms"<<endl;
        init(a,n);
        //静态线程 +barrier 同步
        QueryPerformanceCounter(&timeStart);
        gauss_static3();
        QueryPerformanceCounter(&timeEnd);
        _Trivial = (timeEnd.QuadPart - timeStart.QuadPart)*1000 / quadpart;
        cout<<"静态线程 +barrier 同步:"<<_Trivial<<"ms"<<endl;
        init(a,n);
        //静态线程行分配
        QueryPerformanceCounter(&timeStart);
        gauss_static_row();
        QueryPerformanceCounter(&timeEnd);
        _Trivial = (timeEnd.QuadPart - timeStart.QuadPart)*1000 / quadpart;
        cout<<"静态线程行分配:"<<_Trivial<<"ms"<<endl;
        init(a,n);
        //静态线程列分配
        QueryPerformanceCounter(&timeStart);
        gauss_static_col();
        QueryPerformanceCounter(&timeEnd);
        _Trivial = (timeEnd.QuadPart - timeStart.QuadPart)*1000 / quadpart;
        cout<<"静态线程列分配:"<<_Trivial<<"ms"<<endl;
        init(a,n);
       //静态线程行分配+SSE
        QueryPerformanceCounter(&timeStart);
        gauss_static_row_sse();
        QueryPerformanceCounter(&timeEnd);
        _Trivial = (timeEnd.QuadPart - timeStart.QuadPart)*1000 / quadpart;
        cout<<"静态线程行分配+SSE:"<<_Trivial<<"ms"<<endl;
        init(a,n);
        //静态线程行分配+AVX
        QueryPerformanceCounter(&timeStart);
        gauss_static_row_avx();
        QueryPerformanceCounter(&timeEnd);
        _Trivial = (timeEnd.QuadPart - timeStart.QuadPart)*1000 / quadpart;
        cout<<"静态线程行分配+AVX:"<<_Trivial<<"ms"<<endl;
        init(a,n);
/*
    long long head, tail, freq;
	double sum_time = 0.0;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);

	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    sum_time += (tail - head) * 1000.0 / freq;
    	cout<<sum_time/double(1.0)<<endl;*/
    	return 0;
}

