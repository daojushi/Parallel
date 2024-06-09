#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <mpi.h>
using  namespace std;
const int N =130; // 矩阵的规模
char subMatrix[N][N + 1];      // 消元子
char eliminatedRows[N][N];    // 被消元行
int bLength = 0;              // 被消元行长度

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

double SGE()
{
    int num_procs; // 进程数量
    int rank;      // 进程的排名
    double start_time, end_time;
     start_time = MPI_Wtime();
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs); // 获取进程数量
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);      // 获取当前进程的排名

    int rows_per_proc = bLength / num_procs; // 每个进程处理的行数
    int start_row = rank * rows_per_proc;    // 当前进程处理的起始行
    int end_row = start_row + rows_per_proc; // 当前进程处理的结束行

    for (int i = start_row; i < end_row; i++) // 对每个被消元行进行处理
    {
        for (int j = N - 1; j >= 0; j--) // 从最后一列开始往前遍历
        {
            if (eliminatedRows[i][j] == 1) // 如果eliminatedRows[i][j]为1，说明该行需要进行消元
            {
                if (subMatrix[j][N] != 0) // 如果subMatrix[j][N]!=0，说明该列存在消元子行，需要执行异或操作进行消元
                {
                    // 对每个元素执行异或操作，更新eliminatedRows[i][k]
                    for (int k = 0; k <= j; k++)
                    {
                        eliminatedRows[i][k] = eliminatedRows[i][k] ^ subMatrix[j][k];
                    }
                }
                else // 如果subMatrix[j][N]==0，说明该列没有消元子行，将eliminatedRows[i][k]复制到subMatrix[j][k]中
                {
                    // 将eliminatedRows[i][k]复制到subMatrix[j][k]中
                    for (int k = 0; k <= j; k++)
                    {
                        subMatrix[j][k] = eliminatedRows[i][k];
                    }
                    subMatrix[j][N] = 1; // 标记该行已经有数据
                    break;              // 跳出循环，继续处理下一个被消元行
                }
            }
        }
    }

    // 同步各进程的结果
    MPI_Barrier(MPI_COMM_WORLD);

    // 汇总消元结果到第一个进程
    if (rank != 0)
    {
        for (int i = start_row; i < end_row; i++)
        {
            MPI_Send(subMatrix[i], N + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
        for (int p = 1; p < num_procs; p++)
        {
            int recv_start_row = p * rows_per_proc;
            int recv_end_row = std::min(recv_start_row + rows_per_proc, bLength);
            for (int i = recv_start_row; i < recv_end_row; i++)
            {
                MPI_Recv(subMatrix[i], N + 1, MPI_CHAR, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    // 同步所有进程
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    return (end_time - start_time) * 1000;
}



int main()
{
cout<<"N is"<<N<<endl;
MPI_Init(NULL, NULL); // 初始化MPI

    int rank; // 进程的排名
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        creatSubMatrix("/home/data/Groebner/1_130_22_8/1.txt");
        creatbxyhMatrix("/home/data/Groebner/1_130_22_8/2.txt");
    }

    //SGE_MPI
    MPI_Bcast(subMatrix, N * (N + 1), MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(eliminatedRows, N * N, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bLength, 1, MPI_INT, 0, MPI_COMM_WORLD);
    double time = 0;
    
        time += SGE();
    if (rank == 0) {
        cout << "SGE is:" << time << "ms" << endl;
   }
    MPI_Finalize(); // 终止MPI


}



