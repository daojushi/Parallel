#include<iostream>
#include<sys/time.h>
#include<arm_neon.h>  
#include<stdlib.h>
using namespace std;

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

void neon_u_gaussian_elimination(float** a, int n) {// 两个都使用 NEON 进行 SIMD 优化的高斯消去算法，未对齐
    float32x4_t vec_t, vec_vaik, vec_vakj, vec_vaij, vec_vx, vec_va;
    for (int k = 0; k < n; k++) {
        vec_t = vmovq_n_f32(a[k][k]);
        int j = 0;
        for (j = k + 1; j  < n-3; j += 4) {
            vec_va = vld1q_f32(&a[k][j]);
            vec_va = vdivq_f32(vec_va, vec_t);
            vst1q_f32(&a[k][j], vec_va);
        }
        for (; j < n; j++) {
            a[k][j] = a[k][j] / a[k][k];
        }
        a[k][k] = 1.0f;
        for (int i = k + 1; i < n; i++) {
            vec_vaik = vmovq_n_f32(a[i][k]);
            for (j = k + 1; j  < n-4; j += 4) {
                vec_vakj = vld1q_f32(&a[k][j]);
                vec_vaij = vld1q_f32(&a[i][j]);
                vec_vx = vmulq_f32(vec_vakj, vec_vaik);
                vec_vaij = vsubq_f32(vec_vaij, vec_vx);
                vst1q_f32(&a[i][j], vec_vaij);
            }
            for (; j < n; j++) {
                a[i][j] -= a[i][k] * a[k][j];
            }
            a[i][k] = 0.0f;
        }
    }
}


void first_neon_u_gaussian_elimination(float** a, int n) {// 只有第一个使用 NEON 进行 SIMD 优化的高斯消去算法，未对齐
    float32x4_t vec_t, vec_va;
    for (int k = 0; k < n; k++) {
        vec_t = vmovq_n_f32(a[k][k]);
        int j = 0;
        for (j = k + 1; j  < n-3; j += 4) {
            vec_va = vld1q_f32(&a[k][j]);
            vec_va = vdivq_f32(vec_va, vec_t);
            vst1q_f32(&a[k][j], vec_va);
        }
        for (; j < n; j++) {
            a[k][j] = a[k][j] / a[k][k];
        }
        a[k][k] = 1.0f;
        for (int i = k + 1; i < n; i++)
		{
			for (j = k + 1; j < n; j++)
			{
				a[i][j] -=  a[i][k]* a[k][j];
			}
			a[i][k] = 0;
		}
    }
}


void second_neon_u_gaussian_elimination(float** a, int n) {// 只有第二个使用 NEON 进行 SIMD 优化的高斯消去算法，未对齐
    float32x4_t vec_vaik, vec_vakj, vec_vaij, vec_vx;
    for (int k = 0; k < n; k++) {
        int j = 0;
		for (j = k + 1; j < n; j++)
		{
			a[k][j] = a[k][j] / a[k][k];
		}
		a[k][k] = 1.0;
        for (int i = k + 1; i < n; i++) {
            vec_vaik = vmovq_n_f32(a[i][k]);
            for (j = k + 1; j  < n-4; j += 4) {
                vec_vakj = vld1q_f32(&a[k][j]);
                vec_vaij = vld1q_f32(&a[i][j]);
                vec_vx = vmulq_f32(vec_vakj, vec_vaik);
                vec_vaij = vsubq_f32(vec_vaij, vec_vx);
                vst1q_f32(&a[i][j], vec_vaij);
            }
            for (; j < n; j++) {
                a[i][j] -= a[i][k] * a[k][j];
            }
            a[i][k] = 0.0f;
        }
    }
}




int main()
{
	int n[10]={100,200,300,400,500,600,700,800,900,1000};
    for(int i=0;i<10;i++)
    {
        float** test = new float*[n[i]];
        for (int j = 0; j < n[i]; j++)
        {
            test[j] = new float[n[i]];
        }
        init(test,n[i]);
        int m=n[i];
        cout<<"N is"<<n[i]<<endl;
        srand(time(0));
        struct timeval beg1,end1,beg2,end2,beg3,end3;
        float time1,time2,time3;
        //两处都并行 
        init(test,m);
        gettimeofday(&beg3,NULL);
        neon_u_gaussian_elimination(test,m);
        gettimeofday(&end3,NULL);
        time3=((long long)1000000*end3.tv_sec+(long long)end3.tv_usec- (long long)1000000*beg3.tv_sec-(long long)beg3.tv_usec);
        cout <<"neon_u_gaussian_elimination is "<<time3/1000 <<" ms"<<endl;
		//第一处并行 
        init(test,m);
        gettimeofday(&beg1,NULL);
        first_neon_u_gaussian_elimination(test,m);
        gettimeofday(&end1,NULL);
        time1=((long long)1000000*end1.tv_sec+(long long)end1.tv_usec- (long long)1000000*beg1.tv_sec-(long long)beg1.tv_usec);
        cout <<"first_neon_u_gaussian_elimination "<< time1/1000<<" ms"<<endl;
        
		//第二处并行 
        init(test,m);
         gettimeofday(&beg2,NULL);
        second_neon_u_gaussian_elimination(test,m);
        gettimeofday(&end2,NULL);
        time2=((long long)1000000*end2.tv_sec+(long long)end2.tv_usec- (long long)1000000*beg2.tv_sec-(long long)beg2.tv_usec);
        cout <<"second_neon_u_gaussian_elimination "<<time2/1000 <<" ms"<<endl;
       

	

    }
}


