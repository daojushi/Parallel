#include <iostream>
#include <windows.h>
#include <stdlib.h>

using namespace std;
int a[1000];
void recursion(int n) {
	if (n == 1)
		return;
	else {
		for (int i = 0; i < n / 2; i++) {
			a[i] += a[n - i - 1];
			n = n / 2;
			recursion(n);
		}
	}
}
#define _for(i,a,b) for( int i=(a);i<(b);++i)

int main()
{
    int b = 10;
    int sum;
    long long head, tail, freq;
    double time;
    cout<<"平凡算法：" <<endl;;
    for(int n = 0; n <= 1000; n += b)
    {
        for(int i = 0; i < n; i++)
            a[i] = i;

        QueryPerformanceFrequency((LARGE_INTEGER *)&freq );
        QueryPerformanceCounter((LARGE_INTEGER *)&head);

        for(int m = 0; m < 1000; m++)
        {
            sum = 0;
            for(int i = 0; i < n; i++)
                sum += a[i];
        }
        QueryPerformanceCounter((LARGE_INTEGER *)&tail );
        cout<<"规模为"<<n<<"的程序运行时间为"<<( tail-head) * 1000.0 / freq<<endl;
        if(n == 100)b = 100;
    }
    b=10;cout<<"多链优化算法："<<endl;
    for(int n = 0; n <= 1000; n += b)
    {
        for(int i = 0; i < n; i++)
            a[i] = i;

        QueryPerformanceFrequency((LARGE_INTEGER *)&freq );
        QueryPerformanceCounter((LARGE_INTEGER *)&head);

        for(int m = 0; m < 1000; m++)
        {
            int sum1=0;int sum2=0;
            for(int i=0;i<n;i+=2)
            {
                sum1+=a[i];
                sum2+=a[i+1];
            }
            sum=sum1+sum2;
        }
        QueryPerformanceCounter((LARGE_INTEGER *)&tail );
        cout<<"规模为"<<n<<"的程序运行时间为"<<( tail-head) * 1000.0 / freq<<endl;
        if(n == 100)b = 100;
    }
    b=10;cout<<"二重循环优化算法："<<endl;
    for(int n = 0; n <= 1000; n += b)
    {
        for(int i = 0; i < n; i++)
            a[i] = i;

        QueryPerformanceFrequency((LARGE_INTEGER *)&freq );
        QueryPerformanceCounter((LARGE_INTEGER *)&head);

        for(int m = 0; m < 1000; m++)
        {
            for(int j=n;j>1;j/=2)
                for(int i=0;i<j/2;i++)
                    a[i]=a[i*2]+a[i*2+1];

        }
        QueryPerformanceCounter((LARGE_INTEGER *)&tail );
        cout<<"规模为"<<n<<"的程序运行时间为"<<( tail-head) * 1000.0 / freq<<endl;
        if(n == 100)b = 100;

    }
    b=10;cout<<"递归函数优化算法："<<endl;
    for(int n = 10; n <= 1000; n += b)
    {
        for(int i = 0; i < n; i++)
            a[i] = i;

        QueryPerformanceFrequency((LARGE_INTEGER *)&freq );
        QueryPerformanceCounter((LARGE_INTEGER *)&head);

        for(int m = 0; m < 1000; m++)
        {
           recursion(n);

        }
        QueryPerformanceCounter((LARGE_INTEGER *)&tail );
        cout<<"规模为"<<n<<"的程序运行时间为"<<( tail-head) * 1000.0 / freq<<endl;
        if(n == 100)b = 100;

    }
     b=10;cout<<"unrolled优化算法："<<endl;
    for(int n = 0; n <= 1000; n += b)
    {
        for(int i = 0; i < n; i++)
            a[i] = i;

        QueryPerformanceFrequency((LARGE_INTEGER *)&freq );
        QueryPerformanceCounter((LARGE_INTEGER *)&head);

        for(int m = 0; m < 1000; m++)
        {
            int sum1=0,sum2=0,sum3=0,sum4=0;
            for (int i = 0; i < n; i += 4) {
			sum1 += a[i];
			sum2 += a[i + 1];
			sum3 += a[i + 2];
			sum4 += a[i + 3];
		}
		sum = sum1 + sum2 + sum3 + sum4;

        }
        QueryPerformanceCounter((LARGE_INTEGER *)&tail );
        cout<<"规模为"<<n<<"的程序运行时间为"<<( tail-head) * 1000.0 / freq<<endl;
        if(n == 100)b = 100;

    }
     b=10;cout<<"宏消灭："<<endl;
    for(int n = 0; n <= 1000; n += b)
    {
        for(int i = 0; i < n; i++)
            a[i] = i;

        QueryPerformanceFrequency((LARGE_INTEGER *)&freq );
        QueryPerformanceCounter((LARGE_INTEGER *)&head);

        for(int m = 0; m < 1000; m++)
        {
            _for(i, 0, n) {
			sum += a[i];
		}

        }
        QueryPerformanceCounter((LARGE_INTEGER *)&tail );
        cout<<"规模为"<<n<<"的程序运行时间为"<<( tail-head) * 1000.0 / freq<<endl;
        if(n == 100)b = 100;

    }
}
