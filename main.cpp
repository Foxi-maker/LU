//#include <stdio.h>
//#include <omp.h>
#include <iostream>

void show_mas(double**, int);

int main(int argc, char* argv[])
{
	int N = 3;
	double min = -100., max = 100.;
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
			A[i][j]= U[i][j] = (double)(rand()) / RAND_MAX * (max - min) + min;

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
			L[i][j] = 0.;
		L[i][i] = 1.;
	}

	for (int i = 0; i < N; i++)
		for (int j = i; j < N; j++)
			L[j][i] = U[j][i] / U[i][i];

	for (int k = 1; k < N; k++)
	{
		for (int i = k - 1; i < N; i++)
			for (int j = i; j < N; j++)
				L[j][i] = U[j][i] / U[i][i];

		for (int i = k; i < N; i++)
			for (int j = k - 1; j < N; j++)
				U[i][j] = U[i][j] - L[i][k - 1] * U[k - 1][j];
	}

	printf("Matrix A:\n");
	show_mas(A, N);
	printf("Matrix L:\n");
	show_mas(L, N);
	printf("Matrix U:\n");
	show_mas(U, N);

	double** B = new double* [N];
	for (int i = 0; i < N; i++)
		B[i] = new double[N];

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			double sum = 0.;
			for (int k = 0; k < N; k++)
				sum += L[i][k] * U[k][j];

			B[i][j] = sum;
		}
	}


	printf("Matrix B:\n");
	show_mas(B, N);
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
