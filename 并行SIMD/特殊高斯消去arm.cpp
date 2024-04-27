#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<sys/time.h> 
#include<arm_neon.h>
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
}}

// NEON高斯消元函数
void NeonGE() {
    uint8x16_t sub_row, eliminated_row; // 定义两个 128 位向量类型的变量 sub_row 和 eliminated_row，用于存储 16 个 8 位整数

    // 处理被消元行
    for (int i = 0; i < bLength; i++) { // 循环处理每个被消元行
        for (int j = N - 1; j >= 0; j--) { // 倒序循环处理每个元素
            if (eliminatedRows[i][j] == 1) { // 如果当前位置为消元位置
                if (subMatrix[j][N] != 0) { // 如果存在消元子行
                    // 有消元子行
                    int k;
                    for (k = 0; k = j-16; k += 16) { // 按照 128 位向量的步长，循环处理每个 128 位的元素
                        eliminated_row = vld1q_u8((uint8_t *)&eliminatedRows[i][k]); // 加载当前被消元行中 16 个元素到 eliminated_row 中
                        sub_row = vld1q_u8((uint8_t *)&subMatrix[j][k]); // 加载消元子行中 16 个元素到 sub_row 中

                        eliminated_row = veorq_u8(eliminated_row, sub_row); // 将 sub_row 和 eliminated_row 中的每个元素做异或操作，并将结果存回 eliminated_row 中

                        vst1q_u8((uint8_t *)&eliminatedRows[i][k], eliminated_row); // 将 eliminated_row 中的 16 个元素存回当前被消元行中
                    }
                    for (; k <= j; k++) { // 处理剩余的元素
                        eliminatedRows[i][k] = abs(eliminatedRows[i][k] - subMatrix[j][k]); // 将消元子行中的元素与当前被消元行的元素做差，并取绝对值后存回当前位置
                    }
                }
                else {
                    // 无消元子行消元提升
                    for (int k = 0; k <= j; k++) { // 将消元子行中的元素复制到当前被消元行中
                        subMatrix[j][k] = eliminatedRows[i][k];
                    }
                    subMatrix[j][N] = 1; // 将消元子行的标记位置置为 1，表示已经提升到被消元行中
                    break; // 跳出内层循环，开始处理下一个被消元行
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
    
    //NEON高斯消元
    gettimeofday(&beg2,NULL);
    NeonGE();
    gettimeofday(&end2,NULL);
    time=((long long)1000000*end2.tv_sec+(long long)end2.tv_usec- (long long)1000000*beg2.tv_sec-(long long)beg2.tv_usec)/1000;
    cout <<"NEONGE is "<<time <<" ms"<<endl;
}
