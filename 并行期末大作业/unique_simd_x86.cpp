#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<immintrin.h>
#include<avxintrin.h>
#include<sys/time.h> //linux time
#include<windows.h> //windows time

//这个程序只做了一次测试样例的程序，每次实验时需修改N的值
using namespace std;
const int N=23045;//矩阵的规模，这个值由测试样例的矩阵列数决定
char subMatrix[N][N+1]; //消元子
char eliminatedRows[N][N]; //被消元行
int bLength; //被消元行长度
void creatSubMatrix(char *fname)
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

void creatbxyhMatrix(char *fname)
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
            if(eliminatedRows[i][j]==1)  // 如果eliminatedRows[i][j]为1，说明该行需要进行消元
            {
                if(subMatrix[j][N]!=0) // 如果subMatrix[j][N]!=0，说明该列存在消元子行，需要执行异或操作进行消元
                {
                    // 对每个元素执行异或操作，更新pb[i][k]
                    for(int k=0; k<=j; k++)
                    {
                        eliminatedRows[i][k] = abs(eliminatedRows[i][k] - subMatrix[j][k]);
                    }
                }
                else // 如果subMatrix[j][N]==0，说明该列没有消元子行，将eliminatedRows[i][k]复制到subMatrix[j][k]中
                {
                    // 将eliminatedRows[i][k]复制到pa[j][k]中
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


void SSESGE() // SSE128 special Gauss elimination
{
    __m128i eliminated_row, sub_row;

    // Process eliminated rows
    for (int row_idx = 0; row_idx < bLength; row_idx++) {
        for (int col_idx = N-1; col_idx >= 0; col_idx--) {
            if (eliminatedRows[row_idx][col_idx] == 1) {
                if (subMatrix[col_idx][N] != 0) { // Check if sub row exists
                    // Sub row exists
                    int k;
                    for (k = 0; k+16 <= col_idx; k += 16) {
                        sub_row = _mm_loadu_si128((__m128i *)&subMatrix[col_idx][k]);
                        eliminated_row = _mm_loadu_si128((__m128i *)&eliminatedRows[row_idx][k]);
                        eliminated_row = _mm_xor_si128(sub_row, eliminated_row);
                        _mm_storeu_si128((__m128i *)&eliminatedRows[row_idx][k], eliminated_row);
                    }
                    for (; k <= col_idx; k++)
                        eliminatedRows[row_idx][k] = abs(eliminatedRows[row_idx][k] - subMatrix[col_idx][k]);
                }
                else {
                    // No sub row elimination boosting
                    for (int k = 0; k <= col_idx; k++)
                        subMatrix[col_idx][k] = eliminatedRows[row_idx][k];
                    subMatrix[col_idx][N] = 1; // Mark this row as having data after boosting
                    break;
                }
            }
        }
    }
}

void AVX256SGE() //AVX256 Special Gauss Elimination
{
    __m256i eliminatedRow, subMatrixRow;

    // Process the eliminated rows
    for(int i=0; i<bLength; i++)
        for(int j=N-1; j>=0; j--)
            if(eliminatedRows[i][j] == 1)
            {
                if(subMatrix[j][N] != 0)
                {
                   // There is an elimination sub-row
                    int k;
                    for(k=0; k+32 <= j; k += 32)
                    {
                        eliminatedRow = _mm256_loadu_si256((__m256i*)&eliminatedRows[i][k]);
                        subMatrixRow = _mm256_loadu_si256((__m256i*)&subMatrix[j][k]);
                        eliminatedRow = _mm256_xor_si256(subMatrixRow, eliminatedRow);
                        _mm256_storeu_si256((__m256i*)&eliminatedRows[i][k], eliminatedRow);
                    }
                    for(; k<=j; k++)
                        eliminatedRows[i][k] = abs(eliminatedRows[i][k] - subMatrix[j][k]);

                }
                else
                {
                    // No elimination sub-row, do elimination promotion
                    for(int k=0; k<=j; k++)
                        subMatrix[j][k] = eliminatedRows[i][k];
                    subMatrix[j][N] = 1; // Mark this row as having data after promotion
                    break;
                }
            }
}

void writeresult(char *fname)
{
    ofstream fout;
    fout.open(fname,ios::out);
    for (int i=0;i<bLength;i++)
        {
            for (int j=N-1;j>=0;j--)
                if(eliminatedRows[i][j]==1)
                    fout<<j<<" ";
            fout<<endl;
        }
    fout.close();
}

int main()
{
	creatSubMatrix("消元子.txt");
	creatbxyhMatrix("被消元行.txt");
	//特殊高斯消去平凡算法
    long long head1, tail1 , freq1 ;
    double time1=0;
	QueryPerformanceFrequency((LARGE_INTEGER *)&freq1 );
	QueryPerformanceCounter((LARGE_INTEGER *)&head1);
    SGE();
    QueryPerformanceCounter((LARGE_INTEGER *)&tail1);
    time1=(tail1-head1)*1000.0/freq1;
    cout<<"特殊高斯消去平凡算法时间为"<<time1<<"ms"<<endl;
    //writeresult("特殊高斯消去平凡算法结果.txt");

    //特殊高斯消去SSE算法
    long long head2, tail2 , freq2 ;
    double time2=0;
	QueryPerformanceFrequency((LARGE_INTEGER *)&freq2 );
	QueryPerformanceCounter((LARGE_INTEGER *)&head2);
    SSESGE();
    QueryPerformanceCounter((LARGE_INTEGER *)&tail2 );
    time2=(tail2-head2)*1000.0/freq2;
    cout<<"特殊高斯消去SSE算法时间为"<<time2<<"ms"<<endl;
    //writeresult("特殊高斯消去SSE算法结果.txt");

     //特殊高斯消去AVX256算法
    long long head3, tail3 , freq3 ;
    double time3=0;
	QueryPerformanceFrequency((LARGE_INTEGER *)&freq3 );
	QueryPerformanceCounter((LARGE_INTEGER *)&head3);
    AVX256SGE();
    QueryPerformanceCounter((LARGE_INTEGER *)&tail3 );
    time3=(tail3-head3)*1000.0/freq3;
    cout<<"特殊高斯消去AVX256算法时间为"<<time3<<"ms"<<endl;
    //writeresult("特殊高斯消AVX256算法结果.txt");
	return 0;
 }

