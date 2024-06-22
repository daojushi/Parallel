#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<immintrin.h>
#include<avxintrin.h>
#include<sys/time.h> //linux time
#include<windows.h> //windows time

//�������ֻ����һ�β��������ĳ���ÿ��ʵ��ʱ���޸�N��ֵ
using namespace std;
const int N=23045;//����Ĺ�ģ�����ֵ�ɲ��������ľ�����������
char subMatrix[N][N+1]; //��Ԫ��
char eliminatedRows[N][N]; //����Ԫ��
int bLength; //����Ԫ�г���
void creatSubMatrix(char *fname)
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

void creatbxyhMatrix(char *fname)
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

void SGE() //�����˹��ȥ
{
    for(int i=0; i<bLength; i++) // ��ÿ������Ԫ�н��д���
    {
        for(int j=N-1; j>=0; j--) // �����һ�п�ʼ��ǰ����
        {
            if(eliminatedRows[i][j]==1)  // ���eliminatedRows[i][j]Ϊ1��˵��������Ҫ������Ԫ
            {
                if(subMatrix[j][N]!=0) // ���subMatrix[j][N]!=0��˵�����д�����Ԫ���У���Ҫִ��������������Ԫ
                {
                    // ��ÿ��Ԫ��ִ��������������pb[i][k]
                    for(int k=0; k<=j; k++)
                    {
                        eliminatedRows[i][k] = abs(eliminatedRows[i][k] - subMatrix[j][k]);
                    }
                }
                else // ���subMatrix[j][N]==0��˵������û����Ԫ���У���eliminatedRows[i][k]���Ƶ�subMatrix[j][k]��
                {
                    // ��eliminatedRows[i][k]���Ƶ�pa[j][k]��
                    for(int k=0; k<=j; k++)
                    {
                        subMatrix[j][k] = eliminatedRows[i][k];
                    }
                    subMatrix[j][N] = 1; // ��Ǹ����Ѿ�������
                    break; // ����ѭ��������������һ������Ԫ��
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
	creatSubMatrix("��Ԫ��.txt");
	creatbxyhMatrix("����Ԫ��.txt");
	//�����˹��ȥƽ���㷨
    long long head1, tail1 , freq1 ;
    double time1=0;
	QueryPerformanceFrequency((LARGE_INTEGER *)&freq1 );
	QueryPerformanceCounter((LARGE_INTEGER *)&head1);
    SGE();
    QueryPerformanceCounter((LARGE_INTEGER *)&tail1);
    time1=(tail1-head1)*1000.0/freq1;
    cout<<"�����˹��ȥƽ���㷨ʱ��Ϊ"<<time1<<"ms"<<endl;
    //writeresult("�����˹��ȥƽ���㷨���.txt");

    //�����˹��ȥSSE�㷨
    long long head2, tail2 , freq2 ;
    double time2=0;
	QueryPerformanceFrequency((LARGE_INTEGER *)&freq2 );
	QueryPerformanceCounter((LARGE_INTEGER *)&head2);
    SSESGE();
    QueryPerformanceCounter((LARGE_INTEGER *)&tail2 );
    time2=(tail2-head2)*1000.0/freq2;
    cout<<"�����˹��ȥSSE�㷨ʱ��Ϊ"<<time2<<"ms"<<endl;
    //writeresult("�����˹��ȥSSE�㷨���.txt");

     //�����˹��ȥAVX256�㷨
    long long head3, tail3 , freq3 ;
    double time3=0;
	QueryPerformanceFrequency((LARGE_INTEGER *)&freq3 );
	QueryPerformanceCounter((LARGE_INTEGER *)&head3);
    AVX256SGE();
    QueryPerformanceCounter((LARGE_INTEGER *)&tail3 );
    time3=(tail3-head3)*1000.0/freq3;
    cout<<"�����˹��ȥAVX256�㷨ʱ��Ϊ"<<time3<<"ms"<<endl;
    //writeresult("�����˹��AVX256�㷨���.txt");
	return 0;
 }

