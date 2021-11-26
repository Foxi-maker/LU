#include <stdio.h>
#include <omp.h>
#include <random>

const double eps = 1.e-5;

void show_mas(double**, int);

void LU_parallel(double**, double**, int);
void LU_block_parallel(double**, double**, int, int);

void check_parallel(double**, double**, double**, int);

int main(int argc, char* argv[])
{
	int N = 4096;
	int num_th = 4;
	int block_size = 32;
	double min = -100., max = 100.;
	double t1, t2;

	double** A = new double* [N];
	for (int i = 0; i < N; i++)
		A[i] = new double[N];

	double** L = new double* [N];
	for (int i = 0; i < N; i++)
		L[i] = new double[N];

	double** U = new double* [N];
	for (int i = 0; i < N; i++)
		U[i] = new double[N];

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			A[i][j] = U[i][j] = (double)(rand()) / RAND_MAX * (max - min) + min;

	//show_mas(A, N);
	//printf("\n");

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			L[i][j] = 0.;
		L[i][i] = 1.;
	}

	t1 = omp_get_wtime();
	LU_block_parallel(L, U, N, block_size);
	t2 = omp_get_wtime();
	printf("Time LU = %lf\n", t2 - t1);

	//t1 = omp_get_wtime();
	//check_parallel(A, L, U, N);
	//t2 = omp_get_wtime();
	//printf("Time LU check=%lf\n", t2 - t1);

	//show_mas(L, N);
	//printf("\n");
	//show_mas(U, N);
	//printf("\n");

	for (int i = 0; i < N; i++)
		delete[] A[i];
	delete[] A;
	for (int i = 0; i < N; i++)
		delete[] L[i];
	delete[] L;
	for (int i = 0; i < N; i++)
		delete[] U[i];
	delete[] U;
}

void show_mas(double** mas, int n)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
			printf("%f ", mas[i][j]);
		printf("\n");
	}
}

void LU_parallel(double** L, double** U, int N)
{
	for (int k = 1; k < N; k++)
	{
#pragma omp parallel for 
		for (int i = k - 1; i < N; i++)
			for (int j = i + 1; j < N; j++)
				L[j][i] = U[j][i] / U[i][i];

#pragma omp parallel for 
		for (int i = k; i < N; i++)
			for (int j = k - 1; j < N; j++)
				U[i][j] = U[i][j] - L[i][k - 1] * U[k - 1][j];
	}
}

void LU_block_parallel(double** L, double** U, int N,  int block_size)
{
	double** A11 = new double* [block_size];
	for (int i = 0; i < block_size; i++)
		A11[i] = new double[block_size];

	double** U12 = new double* [block_size];
	for (int i = 0; i < block_size; i++)
		U12[i] = new double[N - block_size];

	double** L21 = new double* [N - block_size];
	for (int i = 0; i < N - block_size; i++)
		L21[i] = new double[block_size];


	for (int bi = 0; bi < N - 1; bi += block_size)
	{
		double temp;
		// Копирование диагонального блока
		for (int i = 0; i < block_size; ++i)
			for (int j = 0; j < block_size; ++j)
				A11[i][j] = U[i + bi][j + bi];

		// Копирование блока U_12
		for (int i = 0; i < block_size; ++i)
			for (int j = 0; j < N - bi - block_size; ++j)
				U12[i][j] = U[i + bi][j + bi + block_size];

		// Копирование блока L_21
		for (int i = 0; i < N - bi - block_size; ++i)
			for (int j = 0; j < block_size; ++j)
				L21[i][j] = U[i + bi + block_size][j + bi];

		// LU разложение диагонального блока
		for (int i = 0; i < block_size - 1; ++i)
		{
#pragma omp parallel for private(temp) 
			for (int j = i + 1; j < block_size; ++j)
			{
				temp = A11[j][i] / A11[i][i];
				for (int k = i + 1; k < block_size; ++k)
				{
					A11[j][k] -= temp * A11[i][k];
				}
				A11[j][i] = temp;
			}
		}
		// Заполнение блока U_12
#pragma omp parallel for private(temp)
		for (int j = 0; j < N - bi - block_size; ++j)
		{
			for (int i = 1; i < block_size; ++i)
			{
				temp = 0.0;
				for (int k = 0; k <= i - 1; ++k)
				{
					temp += A11[i][k] * U12[k][j];
				}
				U12[i][j] -= temp;
			}
		}

		// Заполнение блока L_21
#pragma omp parallel for private(temp) 
		for (int i = 0; i < N - bi - block_size; ++i) {
			for (int j = 0; j < block_size; ++j) {
				temp = 0.0;
				for (int k = 0; k <= j - 1; ++k) {
					temp += L21[i][k] * A11[k][j];
				}
				L21[i][j] = (L21[i][j] - temp) / A11[j][j];
			}
		}

		// Вычисление A22~
#pragma omp parallel for private(temp) 
		for (int i = 0; i < N - bi - block_size; ++i) {
			for (int j = 0; j < N - bi - block_size; ++j) {
				temp = 0.0;
				for (int k = 0; k < block_size; ++k) {
					temp += L21[i][k] * U12[k][j];
				}
				U[i + bi + block_size][j + bi + block_size] -= temp;
			}
		}

		// Перенос лок. массивов в матрицу
		// Диаг. блок
		for (int i = 0; i < block_size; ++i)
			for (int j = i; j < block_size; ++j)
				U[i + bi][j + bi] = A11[i][j];

		for (int i = 1; i < block_size; ++i)
			for (int j = 0; j < i; ++j)
			{
				L[i + bi][j + bi] = A11[i][j];
				U[i + bi][j + bi] = 0.;
			}

		// Блок U_12
		for (int i = 0; i < N - bi - block_size; ++i)
			for (int j = 0; j < block_size; ++j)
				U[j + bi][i + bi + block_size] = U12[j][i];

		// Блок L_21
		for (int i = 0; i < N - bi - block_size; ++i)
			for (int j = 0; j < block_size; ++j)
			{
				L[i + bi + block_size][j + bi] = L21[i][j];
				U[i + bi + block_size][j + bi] = 0.;
			}
	}

	for (int i = 0; i < block_size; i++)
		delete[] A11[i];
	delete[] A11;

	for (int i = 0; i < block_size; i++)
		delete[] U12[i];
	delete[] U12;

	for (int i = 0; i < N - block_size; i++)
		delete[] L21[i];
	delete[] L21;
}

void check_parallel(double** A, double** L, double** U, int N)
{
	double** B = new double* [N];
	for (int i = 0; i < N; i++)
		B[i] = new double[N];

	int check = 0;

#pragma omp parallel for 
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			B[i][j] = 0.;
			for (int k = 0; k < N; k++)
				B[i][j] += L[i][k] * U[k][j];
			if (B[i][j] - A[i][j] > eps)
				check++;
		}
	}

	if (check)
		printf("Mistake!\n");
	else
		printf("Nice!\n");

	for (int i = 0; i < N; i++)
		delete[] B[i];
	delete[] B;
}

void check(double** A, double** L, double** U, int N)
{
	double** B = new double* [N];
	for (int i = 0; i < N; i++)
		B[i] = new double[N];

	int check = 0;

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			B[i][j] = 0.;
			for (int k = 0; k < N; k++)
				B[i][j] += L[i][k] * U[k][j];
			if (B[i][j] - A[i][j] > eps)
			{
				check++;
				printf("%i, %i\n", i, j);
			}

		}
	}

	if (check)
		printf("Mistake!\n");
	else
		printf("Nice!\n");

	for (int i = 0; i < N; i++)
		delete[] B[i];
	delete[] B;
}