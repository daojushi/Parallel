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
float** aligned_init(int n) {
	float** m = (float**)aligned_alloc(32 * n * sizeof(float**), 32);
	for (int i = 0; i < n; i++)
	{
		m[i] = (float*)aligned_alloc(32 * n * sizeof(float*), 32);
	}
	init(m, n);
	return m;
}

void Trivial(float** a, int n) {  // 平凡高斯消去算法
	for (int k = 0; k < n; k++)
	{
		for (int j = k + 1; j < n; j ++)
		{
			a[k][j] = a[k][j] / a[k][k];
		}
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			for (int j = k + 1; j < n; j++)
			{
				a[i][j] -= a[i][k] * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}
void neon_gaussian_elimination(float** matrix, int n) {//使用neon进行SIMD优化的高斯消去算法，对齐
    float32x4_t vec_a, vec_t, vec_aik, vec_akj, vec_aij, vec_x;
    alignas(16) float temp1, temp2;
    for (int k = 0; k < n; k++) {
        vec_t = vdupq_n_f32(matrix[k][k]);
        temp1 = matrix[k][k];
        int j = 0;
        for (j = k + 1; j + 4 < n; j += 4) {
            if (j % 4 != 0) {
                matrix[k][j] = matrix[k][j] / temp1;
                j -= 3;
                continue;
            }
            vec_a = vld1q_f32(&matrix[k][j]);
            vec_a = vdivq_f32(vec_a, vec_t);
            vst1q_f32(&matrix[k][j], vec_a);
        }
        for (; j < n; j++) {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
        matrix[k][k] = 1.0;
        for (int i = k + 1; i < n; i++) {
            vec_aik = vdupq_n_f32(matrix[i][k]);
            temp2 = matrix[i][k];
            for (j = k + 1; j  < n-4; j += 4) {
                if (j % 4 != 0) {
                    matrix[i][j] -= temp2 * matrix[k][j];
                    j -= 3;
                    continue;
                }
                vec_akj = vld1q_f32(&matrix[k][j]);
                vec_aij = vld1q_f32(&matrix[i][j]);
                vec_x = vmulq_f32(vec_akj, vec_aik);
                vec_aij = vsubq_f32(vec_aij, vec_x);
                vst1q_f32(&matrix[i][j], vec_aij);
            }
            for (; j < n; j++) {
                matrix[i][j] -= matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
} //__attribute__((aligned(16)))//确保函数在内存中的对齐方式为 16 字节对齐


void neon_u_gaussian_elimination(float** a, int n) {// 使用 NEON 进行 SIMD 优化的高斯消去算法，未对齐
    float32x4_t vec_t, vec_vaik, vec_vakj, vec_vaij, vec_vx, vec_va;
    for (int k = 0; k < n; k++) {
        vec_t = vmovq_n_f32(a[k][k]);
        int j = 0;
        for (j = k + 1; j + 3 < n; j += 4) {
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

void neon_u_gaussian_elimination_fmsq(float** matrix, int size) { // 使用 NEON 进行 SIMD 优化的高斯消去算法，未对齐，使用 fmsq
    float32x4_t vecA, vecVT, vecAik, vecAkj, vecAij;
    for (int k = 0; k < size; k++) {
        vecVT = vmovq_n_f32(matrix[k][k]); 
        int j = 0;
        for (j = k + 1; j + 4 < size; j += 4) {
            vecA = vld1q_f32(&matrix[k][j]);
            vecA = vdivq_f32(vecA, vecVT);
            vst1q_f32((float32_t*)&matrix[k][j], vecA);
        }
        for (j; j < size; j++) {
            matrix[k][j] = matrix[k][j] / matrix[k][k]; 
        }
        matrix[k][k] = 1.0;
        for (int i = k + 1; i < size; i++) {
            vecAik = vmovq_n_f32(matrix[i][k]);
            for (j = k + 1; j + 4 < size; j += 4) {
                vecAkj = vld1q_f32(&matrix[k][j]);
                vecAij = vld1q_f32(&matrix[i][j]);
                vecAij = vfmsq_f32(vecAij, vecAkj, vecAik);
                vst1q_f32((float32_t*)&matrix[i][j], vecAij);
            }
            for (j; j < size; j++) {
                matrix[i][j] -= matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}

void neon_U_GE_Opt2(float** matrix, int size) {//流水线 
    for (int k = 0; k < size; k++) {
        // 加载重复值到向量中
        float32x4_t vecVT1 = vmovq_n_f32(matrix[k][k]);
        float32x4_t vecVT2 = vmovq_n_f32(matrix[k][k]);

        // 对列向量进行向量化，对向量进行除法运算，对列向量进行还原
        int j = 0;
        for (j = k + 1; j + 8 <= size; j += 8) {
            float32x4_t vecA1 = vld1q_f32(&matrix[k][j]);
            float32x4_t vecA2 = vld1q_f32(&matrix[k][j + 4]);

            vecA1 = vdivq_f32(vecA1, vecVT1);
            vecA2 = vdivq_f32(vecA2, vecVT2);

            vst1q_f32(&matrix[k][j], vecA1);
            vst1q_f32(&matrix[k][j + 4], vecA2);
        }

        // 对于列向量的剩余部分进行标量操作
        for (; j < size; j++) {
            matrix[k][j] /= matrix[k][k];
        }
        matrix[k][k] = 1.0;

        // 对行向量进行向量化并进行流水线优化
        for (int i = k + 1; i < size; i++) {
            float32x4_t vecAik1 = vmovq_n_f32(matrix[i][k]);
            float32x4_t vecAik2 = vmovq_n_f32(matrix[i][k]);

            j = 0;
            for (j = k + 1; j + 8 <= size; j += 8) {
                float32x4_t vecAkj1 = vld1q_f32(&matrix[k][j]);
                float32x4_t vecAkj2 = vld1q_f32(&matrix[k][j + 4]);

                float32x4_t vecAij1 = vld1q_f32(&matrix[i][j]);
                float32x4_t vecAij2 = vld1q_f32(&matrix[i][j + 4]);

                vecAij1 = vfmsq_f32(vecAij1, vecAkj1, vecAik1);
                vecAij2 = vfmsq_f32(vecAij2, vecAkj2, vecAik2);

                vst1q_f32(&matrix[i][j], vecAij1);
                vst1q_f32(&matrix[i][j + 4], vecAij2);
            }

            // 对于行向量的剩余部分进行标量操作
            for (; j < size; j++) {
                matrix[i][j] -= matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
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
        struct timeval beg1,end1,beg2,end2,beg3,end3,beg4,end4,beg5,end5;
        float time;
		//平凡算法 
        init(test,m);
        gettimeofday(&beg1,NULL);
        Trivial(test,m);
        gettimeofday(&end1,NULL);
        time=((long long)1000000*end1.tv_sec+(long long)end1.tv_usec- (long long)1000000*beg1.tv_sec-(long long)beg1.tv_usec)/1000;
        cout <<"gaussian_elimination is "<< time<<" ms"<<endl;
        
		//Neon，对齐 
        //test=aligned_init(m);
        init(test,m);
         gettimeofday(&beg2,NULL);
        neon_gaussian_elimination(test,m);
        gettimeofday(&end2,NULL);
        time=((long long)1000000*end2.tv_sec+(long long)end2.tv_usec- (long long)1000000*beg2.tv_sec-(long long)beg2.tv_usec)/1000;
        cout <<"neon_gaussian_elimination is "<<time <<" ms"<<endl;
       
		//Neon，不对齐
        init(test,m);
        gettimeofday(&beg3,NULL);
        neon_u_gaussian_elimination(test,m);
        gettimeofday(&end3,NULL);
        time=((long long)1000000*end3.tv_sec+(long long)end3.tv_usec- (long long)1000000*beg3.tv_sec-(long long)beg3.tv_usec)/1000;
        cout <<"neon_u_gaussian_elimination is "<<time <<" ms"<<endl;
	
		//Neon，不对齐，使用fmsq
		init(test,m);
		 gettimeofday(&beg4,NULL);
        neon_u_gaussian_elimination_fmsq(test,m);
        gettimeofday(&end4,NULL);
        time=((long long)1000000*end4.tv_sec+(long long)end4.tv_usec- (long long)1000000*beg4.tv_sec-(long long)beg4.tv_usec)/1000;
        cout <<"neon_u_gaussian_elimination_fmsq is "<< time<<" ms"<<endl;
	
		//Neon，不对齐，流水线 
		init(test,m);
		 gettimeofday(&beg5,NULL);
        neon_U_GE_Opt2(test,m);
        gettimeofday(&end5,NULL);
        time= ((long long)1000000*end5.tv_sec+(long long)end5.tv_usec- (long long)1000000*beg5.tv_sec-(long long)beg5.tv_usec)/1000;
        cout <<"Neon_U_GE_Opt2 is "<<time<<" ms"<<endl<<endl;
    }
}


