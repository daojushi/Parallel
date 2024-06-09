#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <mpi.h>
using  namespace std;
const int N =130; // ����Ĺ�ģ
char subMatrix[N][N + 1];      // ��Ԫ��
char eliminatedRows[N][N];    // ����Ԫ��
int bLength = 0;              // ����Ԫ�г���

void creatSubMatrix(char const *fname)
{
    // ������Ԫ�Ӿ���
    ifstream fin;
    fin.open(fname, ios::in);
    if (!fin.is_open()) // ���ļ�ʧ��
    {
        cout << "Failed to open file." << endl;
        exit(0);
    }
    else // ���ļ��ɹ�
    {
        string strTemp;
        while (getline(fin, strTemp)) // ��ȡ�ļ���ÿһ��
        {
            int row_idx, col_idx,first=1;
            istringstream isTemp(strTemp); // ��������ַ���ת��Ϊ������
            while (isTemp >> col_idx) // ����������������
            {
                if(first)//����ǵ�һ��
                {
                    row_idx=col_idx;//��һ��Ϊ������
                    subMatrix[row_idx][N] = 1;//�ھ����б�Ǹ��е�ĩβλ��Ϊ 1
                    first=0;
                }
                subMatrix[row_idx][col_idx] = 1; // ����Ԫ�Ӿ����б�Ǹ�Ԫ��
            }

        }
        fin.close(); // �ر��ļ�
    }
}

void creatbxyhMatrix(char const *fname)
{
    //��������Ԫ�о���
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
            getline(fin, strTemp);// ��ȡ�ļ���ÿһ��
            istringstream isTemp(strTemp); // ��������ַ���ת��Ϊ������
            while (isTemp >> col_idx) // �����������е�������
            {
                eliminatedRows[bLength][col_idx] = 1; // �ڱ���Ԫ�о����б�Ǹ�Ԫ��
            }
            bLength++;
        }
        bLength--; // ���һ�п����ǿ��У������Ҫ��ȥһ��
        fin.close(); // �ر��ļ�

    }
}

double SGE()
{
    int num_procs; // ��������
    int rank;      // ���̵�����
    double start_time, end_time;
     start_time = MPI_Wtime();
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs); // ��ȡ��������
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);      // ��ȡ��ǰ���̵�����

    int rows_per_proc = bLength / num_procs; // ÿ�����̴��������
    int start_row = rank * rows_per_proc;    // ��ǰ���̴������ʼ��
    int end_row = start_row + rows_per_proc; // ��ǰ���̴���Ľ�����

    for (int i = start_row; i < end_row; i++) // ��ÿ������Ԫ�н��д���
    {
        for (int j = N - 1; j >= 0; j--) // �����һ�п�ʼ��ǰ����
        {
            if (eliminatedRows[i][j] == 1) // ���eliminatedRows[i][j]Ϊ1��˵��������Ҫ������Ԫ
            {
                if (subMatrix[j][N] != 0) // ���subMatrix[j][N]!=0��˵�����д�����Ԫ���У���Ҫִ��������������Ԫ
                {
                    // ��ÿ��Ԫ��ִ��������������eliminatedRows[i][k]
                    for (int k = 0; k <= j; k++)
                    {
                        eliminatedRows[i][k] = eliminatedRows[i][k] ^ subMatrix[j][k];
                    }
                }
                else // ���subMatrix[j][N]==0��˵������û����Ԫ���У���eliminatedRows[i][k]���Ƶ�subMatrix[j][k]��
                {
                    // ��eliminatedRows[i][k]���Ƶ�subMatrix[j][k]��
                    for (int k = 0; k <= j; k++)
                    {
                        subMatrix[j][k] = eliminatedRows[i][k];
                    }
                    subMatrix[j][N] = 1; // ��Ǹ����Ѿ�������
                    break;              // ����ѭ��������������һ������Ԫ��
                }
            }
        }
    }

    // ͬ�������̵Ľ��
    MPI_Barrier(MPI_COMM_WORLD);

    // ������Ԫ�������һ������
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

    // ͬ�����н���
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    return (end_time - start_time) * 1000;
}



int main()
{
cout<<"N is"<<N<<endl;
MPI_Init(NULL, NULL); // ��ʼ��MPI

    int rank; // ���̵�����
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
    MPI_Finalize(); // ��ֹMPI


}



