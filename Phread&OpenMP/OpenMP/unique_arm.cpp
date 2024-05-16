#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<sys/time.h>
#include <omp.h>
using namespace std;
const int N=2362;//矩阵的规模
char subMatrix[N][N+1]; //消元子
char eliminatedRows[N][N]; //被消元行
int bLength; //被消元行长度
void creatSubMatrix(char const *fname)
{
    // 创建消元子矩阵
    ifstream fin;
    fin.open(fname, ios::in);
    if (!fin.is_open()) // 打开文件失败
    {
        cout << "Failed to open file." << endl;
        exit(0);
    }
    else // 打开文件成功
    {
        string strTemp;
        while (getline(fin, strTemp)) // 读取文件的每一行
        {
            int row_idx, col_idx,first=1;
            istringstream isTemp(strTemp); // 将读入的字符串转换为输入流
            while (isTemp >> col_idx) // 遍历该行所有索引
            {
                if(first)//如果是第一个
                {
                    row_idx=col_idx;//第一个为行索引
                    subMatrix[row_idx][N] = 1;//在矩阵中标记该行的末尾位置为 1
                    first=0;
                }
                subMatrix[row_idx][col_idx] = 1; // 在消元子矩阵中标记该元素
            }

        }
        fin.close(); // 关闭文件
    }
}

void creatbxyhMatrix(char const *fname)
{
    //创建被消元行矩阵
    ifstream fin;
    fin.open(fname, ios::in);
    if (!fin.is_open())
    {
        cout << "Failed to open file." << endl;
        exit(0);
    }
    else
    {
        string strTemp;
        while (!fin.eof())
        {
            int col_idx;
            getline(fin, strTemp);// 读取文件的每一行
            istringstream isTemp(strTemp); // 将读入的字符串转换为输入流
            while (isTemp >> col_idx) // 遍历该行所有的列索引
            {
                eliminatedRows[bLength][col_idx] = 1; // 在被消元行矩阵中标记该元素
            }
            bLength++;
        }
        bLength--; // 最后一行可能是空行，因此需要减去一行
        fin.close(); // 关闭文件

    }
}

void SGE() //特殊高斯消去
{
    for(int i=0; i<bLength; i++) // 对每个被消元行进行处理
    {
        for(int j=N-1; j>=0; j--) // 从最后一列开始往前遍历
        {
            if(eliminatedRows[i][j]==1);  // 如果void SGE() //特殊高斯消去
		}
    for(int i=0; i<bLength; i++) // 对每个被消元行进行处理
    {
        for(int j=N-1; j>=0; j--) // 从最后一列开始往前遍历
        {
            if(eliminatedRows[i][j]==1)  // 如果eliminatedRows[i][j]为1，说明该行需要进行消元
            {
                if(subMatrix[j][N]!=0) // 如果subMatrix[j][N]!=0，说明该列存在消元子行，需要执行异或操作进行消元
                {
                    // 对每个元素执行异或操作，更新eliminatedRows[i][k]
                    for(int k=0; k<=j; k++)
                    {
                        eliminatedRows[i][k] = abs(eliminatedRows[i][k] - subMatrix[j][k]);
                    }
                }
                else // 如果subMatrix[j][N]==0，说明该列没有消元子行，将eliminatedRows[i][k]复制到subMatrix[j][k]中
                {
                    // 将eliminatedRows[i][k]复制到subMatrix[j][k]中
                    for(int k=0; k<=j; k++)
                    {
                        subMatrix[j][k] = eliminatedRows[i][k];
                    }
                    subMatrix[j][N] = 1; // 标记该行已经有数据
                    break; // 跳出循环，继续处理下一个被消元行
                }
            }
        }
    }
	}
}
void SGE1()
{
    #pragma omp parallel for schedule(dynamic) shared(eliminatedRows, subMatrix, bLength)
    for(int i=0; i<bLength; i++) // 对每个被消元行进行处理
    {
        for(int j=N-1; j>=0; j--) // 从最后一列开始往前遍历
        {
            if(eliminatedRows[i][j]==1)  // 如果eliminatedRows[i][j]为1，说明该行需要进行消元
            {
                if(subMatrix[j][N]!=0) // 如果subMatrix[j][N]!=0，说明该列存在消元子行，需要执行异或操作进行消元
                {
                    // 对每个元素执行异或操作，更新eliminatedRows[i][k]
                    //#pragma omp atomic
                    #pragma omp critical
                    for(int k=0; k<=j; k++)
                    {
                        eliminatedRows[i][k] = abs(eliminatedRows[i][k] - subMatrix[j][k]);
                    }
                }
                else // 如果subMatrix[j][N]==0，说明该列没有消元子行，将eliminatedRows[i][k]复制到subMatrix[j][k]中
                {
                    // 将eliminatedRows[i][k]复制到subMatrix[j][k]中
                    for(int k=0; k<=j; k++)
                    {
                        subMatrix[j][k] = eliminatedRows[i][k];
                    }
                    subMatrix[j][N] = 1; // 标记该行已经有数据
                    break; // 跳出循环，继续处理下一个被消元行
                }
            }
        }
    }
}
int main()
{
	creatSubMatrix("/home/data/Groebner/5_2362_1226_453/1.txt");
	creatbxyhMatrix("/home/data/Groebner/5_2362_1226_453/2.txt");
	srand(time(0));
    struct timeval beg1,end1,beg2,end2;
    float time;
    //SGE
    gettimeofday(&beg1,NULL);
    SGE();
    gettimeofday(&end1,NULL);
    time=((long long)1000000*end1.tv_sec+(long long)end1.tv_usec- (long long)1000000*beg1.tv_sec-(long long)beg1.tv_usec);
    cout <<"SGE is "<< time/1000<<" ms"<<endl;
    //SGE_OpenMP
    gettimeofday(&beg1,NULL);
    SGE1();
    gettimeofday(&end1,NULL);
    time=((long long)1000000*end1.tv_sec+(long long)end1.tv_usec- (long long)1000000*beg1.tv_sec-(long long)beg1.tv_usec);
    cout <<"SGE_OpenMP is "<< time/1000<<" ms"<<endl;

}

