#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<sys/time.h> 
#include<arm_neon.h>
using namespace std;
const int N=2362;//����Ĺ�ģ
char subMatrix[N][N+1]; //��Ԫ��
char eliminatedRows[N][N]; //����Ԫ��
int bLength; //����Ԫ�г���
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

void SGE() //�����˹��ȥ
{
    for(int i=0; i<bLength; i++) // ��ÿ������Ԫ�н��д���
    {
        for(int j=N-1; j>=0; j--) // �����һ�п�ʼ��ǰ����
        {
            if(eliminatedRows[i][j]==1);  // ���void SGE() //�����˹��ȥ
}
    for(int i=0; i<bLength; i++) // ��ÿ������Ԫ�н��д���
    {
        for(int j=N-1; j>=0; j--) // �����һ�п�ʼ��ǰ����
        {
            if(eliminatedRows[i][j]==1)  // ���eliminatedRows[i][j]Ϊ1��˵��������Ҫ������Ԫ
            {
                if(subMatrix[j][N]!=0) // ���subMatrix[j][N]!=0��˵�����д�����Ԫ���У���Ҫִ��������������Ԫ
                {
                    // ��ÿ��Ԫ��ִ��������������eliminatedRows[i][k]
                    for(int k=0; k<=j; k++)
                    {
                        eliminatedRows[i][k] = abs(eliminatedRows[i][k] - subMatrix[j][k]);
                    }
                }
                else // ���subMatrix[j][N]==0��˵������û����Ԫ���У���eliminatedRows[i][k]���Ƶ�subMatrix[j][k]��
                {
                    // ��eliminatedRows[i][k]���Ƶ�subMatrix[j][k]��
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
}}

// NEON��˹��Ԫ����
void NeonGE() {
    uint8x16_t sub_row, eliminated_row; // �������� 128 λ�������͵ı��� sub_row �� eliminated_row�����ڴ洢 16 �� 8 λ����

    // ������Ԫ��
    for (int i = 0; i < bLength; i++) { // ѭ������ÿ������Ԫ��
        for (int j = N - 1; j >= 0; j--) { // ����ѭ������ÿ��Ԫ��
            if (eliminatedRows[i][j] == 1) { // �����ǰλ��Ϊ��Ԫλ��
                if (subMatrix[j][N] != 0) { // ���������Ԫ����
                    // ����Ԫ����
                    int k;
                    for (k = 0; k = j-16; k += 16) { // ���� 128 λ�����Ĳ�����ѭ������ÿ�� 128 λ��Ԫ��
                        eliminated_row = vld1q_u8((uint8_t *)&eliminatedRows[i][k]); // ���ص�ǰ����Ԫ���� 16 ��Ԫ�ص� eliminated_row ��
                        sub_row = vld1q_u8((uint8_t *)&subMatrix[j][k]); // ������Ԫ������ 16 ��Ԫ�ص� sub_row ��

                        eliminated_row = veorq_u8(eliminated_row, sub_row); // �� sub_row �� eliminated_row �е�ÿ��Ԫ���������������������� eliminated_row ��

                        vst1q_u8((uint8_t *)&eliminatedRows[i][k], eliminated_row); // �� eliminated_row �е� 16 ��Ԫ�ش�ص�ǰ����Ԫ����
                    }
                    for (; k <= j; k++) { // ����ʣ���Ԫ��
                        eliminatedRows[i][k] = abs(eliminatedRows[i][k] - subMatrix[j][k]); // ����Ԫ�����е�Ԫ���뵱ǰ����Ԫ�е�Ԫ�������ȡ����ֵ���ص�ǰλ��
                    }
                }
                else {
                    // ����Ԫ������Ԫ����
                    for (int k = 0; k <= j; k++) { // ����Ԫ�����е�Ԫ�ظ��Ƶ���ǰ����Ԫ����
                        subMatrix[j][k] = eliminatedRows[i][k];
                    }
                    subMatrix[j][N] = 1; // ����Ԫ���еı��λ����Ϊ 1����ʾ�Ѿ�����������Ԫ����
                    break; // �����ڲ�ѭ������ʼ������һ������Ԫ��
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
    time=((long long)1000000*end1.tv_sec+(long long)end1.tv_usec- (long long)1000000*beg1.tv_sec-(long long)beg1.tv_usec)/1000;
    cout <<"SGE is "<< time<<" ms"<<endl;
    
    //NEON��˹��Ԫ
    gettimeofday(&beg2,NULL);
    NeonGE();
    gettimeofday(&end2,NULL);
    time=((long long)1000000*end2.tv_sec+(long long)end2.tv_usec- (long long)1000000*beg2.tv_sec-(long long)beg2.tv_usec)/1000;
    cout <<"NEONGE is "<<time <<" ms"<<endl;
}
