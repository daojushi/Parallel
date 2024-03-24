#include <iostream>
#include <windows.h>
#include <stdlib.h>

using namespace std;
int matrix[1000][1000];
int vectors[1000];
int sum[1000];
int sum2[1000];
int main()
{

    int a = 10;
    long long head, tail , freq ;
    for(int i = 0; i < 1000; i++)
        for(int j = 0; j < 1000; j++)
            matrix[i][j] = i + j;
    for(int i = 0; i < 1000; i++)
        vectors[i] = i;
    cout<<"平凡算法："<<endl;
    for(int n = 0; n <= 1000; n += a)
    {


        QueryPerformanceFrequency((LARGE_INTEGER *)&freq );
        QueryPerformanceCounter((LARGE_INTEGER *)&head);

        for(int m = 0; m < 1000; m++)
        {
            for(int i = 0; i < n; i++)
            {
                sum[i] = 0;
                for(int j = 0; j < n; j++)
                    sum[i] += matrix[j][i] * vectors[j];
            }

        }
        QueryPerformanceCounter((LARGE_INTEGER *)&tail );

        cout<<"规模为"<<n<<"的程序运行时间为："<< ( tail - head) * 1000.0 / freq<<endl;
        if(n >= 100) a = 100;
    }
    a=10;
     cout<<"优化算法："<<endl;
     for(int n = 0; n <= 1000; n += a)
    {


        QueryPerformanceFrequency((LARGE_INTEGER *)&freq );
        QueryPerformanceCounter((LARGE_INTEGER *)&head);

        for(int m = 0; m < 1000; m++)
        {
            for(int i = 0; i < n; i++)
            {
                sum2[i] = 0;

            }
            for(int j=0;j<n;j++)
                for(int i=0;i<n;i++)
                    sum2[i]+=matrix[j][i]*vectors[j];

        }
        QueryPerformanceCounter((LARGE_INTEGER *)&tail );

        cout<<"规模为"<<n<<"的程序运行时间为："<< ( tail - head) * 1000.0 / freq<<endl;
        if(n >= 100) a = 100;
    }
     a=10;
     cout<<"优化+unrolled算法："<<endl;
     for(int n = 0; n <= 1000; n += a)
    {


        QueryPerformanceFrequency((LARGE_INTEGER *)&freq );
        QueryPerformanceCounter((LARGE_INTEGER *)&head);

        for(int m = 0; m < 1000; m++)
        {
            for (int i = 0; i < n; i++)
				sum2[i] = 0;
			for (int i = 0; i < n; i++)
				for (int j = 0; j < n; j++)
				{
					sum2[j] += matrix[i][j] * vectors[i];
					j++;
					sum2[j] += matrix[i][j] * vectors[i];
				}

        }
        QueryPerformanceCounter((LARGE_INTEGER *)&tail );

        cout<<"规模为"<<n<<"的程序运行时间为："<< ( tail - head) * 1000.0 / freq<<endl;
        if(n >= 100) a = 100;
    }
}
