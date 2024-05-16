#include<iostream>
#include<sys/time.h>
#include<arm_neon.h>
#include<stdlib.h>
#include<pthread.h>
#include<semaphore.h>
#include<unistd.h>
using namespace std;
const int n=100;
float a[n][n];
const int NUM_THREADS=7;
int id[NUM_THREADS];
typedef struct{
	int k;
	int t_id;
}threadParam_t;
sem_t sem_main;
sem_t sem_workerstart[NUM_THREADS];
sem_t sem_workerend[NUM_THREADS];
pthread_barrier_t childbarrier_row;
pthread_barrier_t childbarrier_col;
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
//静态线程行分配+neon
void * dealwithbyrow_Neon(void * ID)
{
    int* threadid= (int*)ID;
    float32x4_t t1,t2,t3;
    float32x2_t s1, s2;
    for(int k=0;k<n;k++)
    {
        int begin=k+1+*threadid*((n-k-1)/NUM_THREADS);
        int end=begin+(n-k-1)/NUM_THREADS;
        if(end>n)
            end=n;
        int preprocessnumber=(n-k-1)%4;
        int begincol=k+1+preprocessnumber;
        for(int i=begin;i<end;i++)
        {
            for(int j=k+1;j<preprocessnumber;j++)
            {
                a[i][j]=a[i][j]-a[i][k]*a[k][j];
            }
            a[i][k]=0;
        }
        for(int i=begin;i<end;i++)
        {
            float head1[4]={a[i][k],a[i][k],a[i][k],a[i][k]};
            t3=vld1q_f32(head1);
            for(int j=begincol;j<n;j+=4)
            {
                t1=vld1q_f32(a[k]+j);
                t2=vld1q_f32(a[i]+j);
                t1=vmulq_f32(t1,t3);
                t2=vsubq_f32(t2,t3);
                s1 = vget_low_f32(t2);
                s2 = vget_high_f32(t2);
                vst1_lane_f32(a[i]+j,s2,0);
                vst1_lane_f32(a[i]+j+2,s1,0);
            }
            a[i][k]=0;
        }
        pthread_barrier_wait(&childbarrier_row);
        sem_post(&sem_main);
    }
    pthread_exit(NULL);
}
void Gausseliminate_pthread_row_Neon()
{
    pthread_t threadID[NUM_THREADS];
    float32x4_t t1,t2,t3;
    float32x2_t s1, s2;
    for(int k=0;k<n;k++)
    {
        int preprocessnumber=(n-k-1)%4;
        int begin=k+1+preprocessnumber;
        float head[4]={a[k][k],a[k][k],a[k][k],a[k][k]};
        t2=vld1q_f32(head);
        for(int j=k+1;j<k+1+preprocessnumber;j++)
        {
            a[k][j]=a[k][j]/a[k][k];
        }
        for(int j=begin;j<n;j++)
        {
            t1=vld1q_f32(a[k]+j);
            t1=vdivq_f32(t1,t2);
            s1 = vget_low_f32(t1);
            s2 = vget_high_f32(t1);
            vst1_lane_f32(a[k]+j,s2,0);
            vst1_lane_f32(a[k]+j+2,s1,0);
        }
        a[k][k]=1;
        if(k==0)
        {
            for(int i=0;i<NUM_THREADS;i++)
            {
                pthread_create(&threadID[i],NULL,dealwithbyrow_Neon,(void*)&id[i]);
            }
        }
        for(int i=0;i<NUM_THREADS;i++)//
        {
            sem_wait(&sem_main);
        }
    }
    for(int i=0;i<NUM_THREADS;i++)
    {
        pthread_join(threadID[i],NULL);
    }
    return;
}
int main()
{
	float** a = new float*[n];
        for (int j = 0; j < n; j++)
        {
            a[j] = new float[n];
        }
    pthread_barrier_init(&childbarrier_row, NULL,NUM_THREADS);
    pthread_barrier_init(&childbarrier_col,NULL, NUM_THREADS);
    sem_init(&sem_main, 0, 0);
	struct timeval beg1,end1;
	srand(time(0));
	float time;
	cout<<"N is"<<n<<endl;
	init(a,n);
    gettimeofday(&beg1,NULL);
    normal(a,n);
    gettimeofday(&end1,NULL);
    time=((long long)1000000*end1.tv_sec+(long long)end1.tv_usec- (long long)1000000*beg1.tv_sec-(long long)beg1.tv_usec);
    cout <<"normal_gaussian_elimination is "<< time/1000<<" ms"<<endl;
    init(a,n);
    gettimeofday(&beg1,NULL);
    gauss_static_row();
    gettimeofday(&end1,NULL);
    time=((long long)1000000*end1.tv_sec+(long long)end1.tv_usec- (long long)1000000*beg1.tv_sec-(long long)beg1.tv_usec);
    cout <<"staticbyrow_gaussian_elimination is "<< time/1000<<" ms"<<endl;
     init(a,n);
    gettimeofday(&beg1,NULL);
   	Gausseliminate_pthread_row_Neon();
    gettimeofday(&end1,NULL);
    time=((long long)1000000*end1.tv_sec+(long long)end1.tv_usec- (long long)1000000*beg1.tv_sec-(long long)beg1.tv_usec);
    cout <<"staticbyrow_NEON_gaussian_elimination is "<< time/1000<<" ms"<<endl;
	return 0;
}
