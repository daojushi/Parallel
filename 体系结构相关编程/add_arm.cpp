#include<iostream>
#include<sys/time.h>

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
    struct timeval t1, t2;
    double time,tt;
    cout<<"平凡算法：" <<endl;;
    for(int n = 0; n <= 1000; n += b)
    {
        for(int i = 0; i < n; i++)
            a[i] = i;

        gettimeofday(&t1, NULL);

        for(int m = 0; m < 50000; m++)
        {
            int sum = 0;
            for(int i = 0; i < n; i++)
                sum += a[i];
        }
        gettimeofday(&t2, NULL);
		tt = (t2.tv_sec - t1.tv_sec) * 1000000.0 + (double)(t2.tv_usec - t1.tv_usec);
        cout<<"规模为"<<n<<"的程序运行时间为"<<tt<<endl;
        if(n == 100)b = 100;
    }
    b=10;cout<<"多链优化算法："<<endl;
    for(int n = 0; n <= 1000; n += b)
    {
        for(int i = 0; i < n; i++)
            a[i] = i;

        gettimeofday(&t1, NULL);

        for(int m = 0; m < 50000; m++)
        {
            int sum1=0;int sum2=0;
            for(int i=0;i<n;i+=2)
            {
                sum1+=a[i];
                sum2+=a[i+1];
            }
            int sum=sum1+sum2;
        }
        gettimeofday(&t2, NULL);
		tt = (t2.tv_sec - t1.tv_sec) * 1000000.0 + (double)(t2.tv_usec - t1.tv_usec);
        cout<<"规模为"<<n<<"的程序运行时间为"<<tt<<endl;
        if(n == 100)b = 100;
    }
    b=10;cout<<"二重循环优化算法："<<endl;
    for(int n = 0; n <= 1000; n += b)
    {
        for(int i = 0; i < n; i++)
            a[i] = i;

        gettimeofday(&t1, NULL);

        for(int m = 0; m < 50000; m++)
        {
            for(int j=n;j>1;j/=2)
                for(int i=0;i<j/2;i++)
                    a[i]=a[i*2]+a[i*2+1];

        }
        gettimeofday(&t2, NULL);
		tt = (t2.tv_sec - t1.tv_sec) * 1000000.0 + (double)(t2.tv_usec - t1.tv_usec);
        cout<<"规模为"<<n<<"的程序运行时间为"<<tt<<endl;
        if(n == 100)b = 100;

    }
    b=10;cout<<"递归函数优化算法："<<endl;
    for(int n = 10; n <= 1000; n += b)
    {
        for(int i = 0; i < n; i++)
            a[i] = i;

        gettimeofday(&t1, NULL);

        for(int m = 0; m < 50000; m++)
        {
           recursion(n);

        }
         
        gettimeofday(&t2, NULL);
		tt = (t2.tv_sec - t1.tv_sec) * 1000000.0 + (double)(t2.tv_usec - t1.tv_usec);
        cout<<"规模为"<<n<<"的程序运行时间为"<<tt<<endl;
        if(n == 100)b = 100;

    }
     b=10;cout<<"unrolled优化算法："<<endl;
    for(int n = 0; n <= 1000; n += b)
    {
        for(int i = 0; i < n; i++)
            a[i] = i;

        gettimeofday(&t1, NULL);
        for(int m = 0; m < 50000; m++)
        {
            int sum1=0,sum2=0,sum3=0,sum4=0;
            for (int i = 0; i < n; i += 4) {
			sum1 += a[i];
			sum2 += a[i + 1];
			sum3 += a[i + 2];
			sum4 += a[i + 3];
		}
		int sum = sum1 + sum2 + sum3 + sum4;

        }
        gettimeofday(&t2, NULL);
		tt = (t2.tv_sec - t1.tv_sec) * 1000000.0 + (double)(t2.tv_usec - t1.tv_usec);
        cout<<"规模为"<<n<<"的程序运行时间为"<<tt<<endl;
        if(n == 100)b = 100;

    }
     b=10;cout<<"宏消灭："<<endl;
    for(int n = 0; n <= 1000; n += b)
    {
        for(int i = 0; i < n; i++)
            a[i] = i;
		int sum;
        gettimeofday(&t1, NULL);

        for(int m = 0; m < 50000; m++)
        {
            _for(i, 0, n) {
			sum += a[i];
		}

        }
         gettimeofday(&t2, NULL);
		tt = (t2.tv_sec - t1.tv_sec) * 1000000.0 + (double)(t2.tv_usec - t1.tv_usec);
        cout<<"规模为"<<n<<"的程序运行时间为"<<tt<<endl;
        if(n == 100)b = 100;

    }
}
