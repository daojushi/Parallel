#include <iostream>
#include <emmintrin.h>
#include<time.h>
#include<Windows.h>
#include <immintrin.h>
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
	float** m = (float**)_aligned_malloc(32 * n * sizeof(float**), 32);
	for (int i = 0; i < n; i++)
	{
		m[i] = (float*)_aligned_malloc(32 * n * sizeof(float*), 32);
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
void cache(float** a, int n) {  // cache优化高斯消去算法
	float x1, x2;
	for (int k = 0; k < n; k++)
	{
		x1 = a[k][k];
		for (int j = k + 1; j < n; j++)
		{
			a[k][j] = a[k][j] / 1;
		}
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			x2 = a[i][k];
			for (int j = k + 1; j < n; j++)
			{
				a[i][j] -= x2 * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}
void sse_gaussian_elimination(float** matrix, int n) {//使用SSE进行SIMD优化的高斯消去算法，对齐
    __m128 vec_a, vec_t, vec_aik, vec_akj, vec_aij, vec_x;
    float temp1,temp2;
    for (int k = 0; k < n; k++) {
        vec_t = _mm_set1_ps(matrix[k][k]);
        temp1 = matrix[k][k];
        int j = 0;
        for (j = k + 1; j < n - 4; j += 4) {
            if (j % 4 != 0) {
                matrix[k][j] = matrix[k][j] / temp1;
                j -= 3;
                continue;
            }
            vec_a = _mm_load_ps(&matrix[k][j]);
            vec_a = _mm_div_ps(vec_a, vec_t);
            _mm_store_ps(&matrix[k][j], vec_a);
        }
        for (; j < n; j++) {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
        matrix[k][k] = 1.0;
        for (int i = k + 1; i < n; i++) {
            vec_aik = _mm_set1_ps(matrix[i][k]);
            temp2 = matrix[i][k];
            for (j = k + 1; j  < n - 4; j += 4) {
                if (j % 4 != 0) {
                    matrix[i][j] -= temp2 * matrix[k][j];
                    j -= 3;
                    continue;
                }
                vec_akj = _mm_load_ps(&matrix[k][j]);
                vec_aij = _mm_load_ps(&matrix[i][j]);
                vec_x = _mm_mul_ps(vec_akj, vec_aik);
                vec_aij = _mm_sub_ps(vec_aij, vec_x);
                _mm_store_ps(&matrix[i][j], vec_aij);
            }
            for (; j < n; j++) {
                matrix[i][j] -= matrix[i][k] * matrix[k][j];
            }
            _mm_store_ss(&matrix[i][k], _mm_load_ss(&matrix[i][k]) - temp2 * _mm_load_ss(&matrix[k][k]));
        }
    }
}
void sse_u_gaussian_elimination(float** matrix, int n) { // 使用SSE进行SIMD优化的高斯消去算法，未对齐
    __m128 vec_a, vec_t, vec_aik, vec_akj, vec_aij, vec_x;
    float temp2;
    for (int k = 0; k < n; k++) {
        vec_t = _mm_set1_ps(matrix[k][k]);
        int j = 0;
        for (j = k + 1; j  < n - 4; j += 4) {
            vec_a = _mm_loadu_ps(&matrix[k][j]);
            vec_a = _mm_div_ps(vec_a, vec_t);
            _mm_storeu_ps(&matrix[k][j], vec_a);
        }
        for (; j < n; j++) {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
        matrix[k][k] = 1.0;
        for (int i = k + 1; i < n; i++) {
            vec_aik = _mm_set1_ps(matrix[i][k]);
            temp2 = matrix[i][k];
            for (j = k + 1; j + 4 < n; j += 4) {
                vec_akj = _mm_loadu_ps(&matrix[k][j]);
                vec_aij = _mm_loadu_ps(&matrix[i][j]);
                vec_x = _mm_mul_ps(vec_akj, vec_aik);
                vec_aij = _mm_sub_ps(vec_aij, vec_x);
                _mm_storeu_ps(&matrix[i][j], vec_aij);
            }
            for (; j < n; j++) {
                matrix[i][j] -= matrix[i][k] * matrix[k][j];
            }
           _mm_store_ss(&matrix[i][k], _mm_load_ss(&matrix[i][k]) - temp2 * _mm_load_ss(&matrix[k][k]));
        }
    }
}


void avx_gaussian_elimination(float** matrix, int n) {//使用AVX进行SIMD优化的高斯消去算法，对齐
    __m256 vec_a, vec_t, vec_aik, vec_akj, vec_aij, vec_x;
    float temp1, temp2;
    for (int k = 0; k < n; k++) {
        vec_t = _mm256_set1_ps(matrix[k][k]);
        temp1 = matrix[k][k];
        int j = 0;
        for (j = k + 1; j < n - 8; j += 8) {
            if (j % 8 != 0) {
                matrix[k][j] = matrix[k][j] / temp1;
                j -= 7;
                continue;
            }
            vec_a = _mm256_load_ps(&matrix[k][j]);
            vec_a = _mm256_div_ps(vec_a, vec_t);
            _mm256_store_ps(&matrix[k][j], vec_a);
        }
        for (; j < n; j++) {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
        matrix[k][k] = 1.0;
        for (int i = k + 1; i < n; i++) {
            vec_aik = _mm256_set1_ps(matrix[i][k]);
            temp2 = matrix[i][k];
            for (j = k + 1; j  < n-8; j += 8) {
                if (j % 8 != 0) {
                    matrix[i][j] -= temp2 * matrix[k][j];
                    j -= 7;
                    continue;
                }
                vec_akj = _mm256_load_ps(&matrix[k][j]);
                vec_aij = _mm256_load_ps(&matrix[i][j]);
                vec_x = _mm256_mul_ps(vec_akj, vec_aik);
                vec_aij = _mm256_sub_ps(vec_aij, vec_x);
                _mm256_store_ps(&matrix[i][j], vec_aij);
            }
            for (; j < n; j++) {
                matrix[i][j] -= matrix[i][k] * matrix[k][j];
            }
           //_mm256_storeu_ps(&matrix[i][k], _mm256_sub_ps(_mm256_loadu_ps(&matrix[i][k]), _mm256_mul_ps(_mm256_set1_ps(temp2), _mm256_loadu_ps(&matrix[k][k]))));
            matrix[i][k]=0;
        }
    }
}
void avx_u_gaussian_elimination(float** matrix, int size) { // 使用AVX指令集进行SIMD优化的高斯消去算法，未对齐
    __m256 vec_kk, vec_ik, vec_kj, vec_ij, vec_x;
    for (int k = 0; k < size; k++) {
        vec_kk = _mm256_set1_ps(matrix[k][k]);
        int j = 0;
        for (j = k + 1; j + 8 < size; j += 8) {
            vec_ik = _mm256_loadu_ps(&matrix[j][k]);
            vec_ik = _mm256_div_ps(vec_ik, vec_kk);
            _mm256_storeu_ps(&matrix[j][k], vec_ik);
        }
        for (j; j < size; j++) {
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        }
        matrix[k][k] = 1.0;
        for (int i = k + 1; i < size; i++) {
            vec_ik = _mm256_set1_ps(matrix[i][k]);
            for (j = k + 1; j + 8 < size; j += 8) {
                vec_kj = _mm256_loadu_ps(&matrix[k][j]);
                vec_ij = _mm256_loadu_ps(&matrix[i][j]);
                vec_x = _mm256_mul_ps(vec_kj, vec_ik);
                vec_ij = _mm256_sub_ps(vec_ij, vec_x);
                _mm256_storeu_ps(&matrix[i][j], vec_ij);
            }
            for (j; j < size; j++) {
                matrix[i][j] -= matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
}


int main()
{
    int n[11]={10,100,200,300,400,500,600,700,800,900,1000};
    for(int i=0;i<11;i++)
    {
        float** test = new float*[n[i]];
        for (int j = 0; j < n[i]; j++)
        {
            test[j] = new float[n[i]];
        }
        init(test,n[i]);
        cout<<"本次实验规模为:"<<n[i]<<endl;
        srand(time(NULL));
        LARGE_INTEGER timeStart;	//开始时间
        LARGE_INTEGER timeEnd;		//结束时间
        LARGE_INTEGER frequency;	//计时器频率
        QueryPerformanceFrequency(&frequency);
        double quadpart = (double)frequency.QuadPart;//计时器频率

        //平凡算法
        QueryPerformanceCounter(&timeStart);
        Trivial(test,n[i]);
        QueryPerformanceCounter(&timeEnd);
        double _Trivial = (timeEnd.QuadPart - timeStart.QuadPart)*1000 / quadpart;
        cout<<"平凡算法:"<<_Trivial<<"ms"<<endl;
        init(test,n[i]);

        //cache优化算法
        QueryPerformanceCounter(&timeStart);
        cache(test,n[i]);
        QueryPerformanceCounter(&timeEnd);
        double _cache = (timeEnd.QuadPart - timeStart.QuadPart)*1000 / quadpart;
        cout<<"cache优化算法:"<<_cache<<"ms"<<endl;
        test=aligned_init(n[i]);

        //sse对齐
        QueryPerformanceCounter(&timeStart);
        sse_gaussian_elimination(test,n[i]);
        QueryPerformanceCounter(&timeEnd);
        double _sse_gaussian_elimination = (timeEnd.QuadPart - timeStart.QuadPart)*1000 / quadpart;
        cout<<"sse对齐算法:"<<_sse_gaussian_elimination<<"ms"<<endl;
        init(test,n[i]);

        //sse不对齐
        QueryPerformanceCounter(&timeStart);
        sse_u_gaussian_elimination(test,n[i]);
        QueryPerformanceCounter(&timeEnd);
        double _sse_u_gaussian_elimination = (timeEnd.QuadPart - timeStart.QuadPart)*1000 / quadpart;
        cout<<"sse不对齐算法:"<<_sse_u_gaussian_elimination<<"ms"<<endl;
        test=aligned_init(n[i]);

        //avx对齐
        QueryPerformanceCounter(&timeStart);
        avx_gaussian_elimination(test,n[i]);
        QueryPerformanceCounter(&timeEnd);
        double _avx_gaussian_elimination = (timeEnd.QuadPart - timeStart.QuadPart)*1000 / quadpart;
        cout<<"avx对齐算法:"<<_avx_gaussian_elimination<<"ms"<<endl;
        init(test,n[i]);

        //avx不对齐
        QueryPerformanceCounter(&timeStart);
        avx_u_gaussian_elimination(test,n[i]);
        QueryPerformanceCounter(&timeEnd);
        double _avx_u_gaussian_elimination = (timeEnd.QuadPart - timeStart.QuadPart)*1000 / quadpart;
        cout<<"avx不对齐算法:"<<_avx_u_gaussian_elimination<<"ms"<<endl;
        init(test,n[i]);

    }
    cout<<"实验结束"<<endl;
    return 0;
}
