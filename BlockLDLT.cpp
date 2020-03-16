//============================================================================
// Name        : BlockLDLT.cpp
// Author      : Tomohiro Suzuki
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C, Ansi-style
//============================================================================

#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <cassert>
#include <ctime>
#include <omp.h>
#include <mkl.h>

using namespace std;

// Generate random LOWER matrix
void Gen_rand_lower_mat(const int m, const int n, double *A)
{
//	srand(time(NULL));
	srand(20200314);

	for (int i=0; i<m; i++)
		for (int j=0; j<n; j++)
			if (i >= j)
				A[i+j*m] = 1.0 - 2*(double)rand() / RAND_MAX;
			else
				A[i+j*m] = 0.0;
}

// Show matrix
void Show_mat(const int m, const int n, double *A)
{
	for (int i=0; i<m; i++) {
		for (int j=0; j<n; j++)
			printf("% 6.4lf, ",A[i + j*m]);
		cout << endl;
	}
	cout << endl;
}

// Debug mode
#define DEBUG

// Trace mode
//#define TRACE

#ifdef TRACE
extern void trace_cpu_start();
extern void trace_cpu_stop(const char *color);
extern void trace_label(const char *color, const char *label);
#endif

int main(const int argc, const char **argv)
{
	// Usage "a.out [size of matrix: m ]"
	assert(argc > 1);

	const int m = atoi(argv[1]);     // # rows and columns <- the matrix is square
	const int lda = m;

	double* A = new double [m*m];    // Original matrix
	int* ipiv = new int[m];          // Pivot vector

	Gen_rand_lower_mat(m,m,A);       // Randomize elements of orig. matrix

	////////// Debug mode //////////
	#ifdef DEBUG
	double *OA = new double[m*m];
	cblas_dcopy(m*m, A, 1, OA, 1);
	for (int i=0; i<m; i++)
		for (int j=0; j<=i; j++)
			OA[j+i*m] = OA[i+j*m];   // Fill the upper triangular part

	double *D = new double[m*m];
	double *L = new double[m*m];
	for (int i=0; i<m*m; i++)        // Initialize D and L
		D[i] = L[i] = 0.0;
	for (int i=0; i<m; i++)
		L[i+i*m] = 1.0;
	#endif
	////////// Debug mode //////////

	Show_mat(m,m,A);

	double timer = omp_get_wtime();    // Timer start

	assert(0 == LAPACKE_dsytrf(MKL_COL_MAJOR, 'L', m, A, lda, ipiv));

	timer = omp_get_wtime() - timer;   // Timer stop

	cout << "m = " << m << ", time = " << timer << endl;

	cout << "LDL(A):\n";
	Show_mat(m,m,A);

	for (int i=0; i<m; i++)
		printf("%3d\n",ipiv[i]);
	printf("\n");

	////////// Debug mode //////////
	#ifdef DEBUG
	// Make L and D
	for (int k=0; k<m; k++)
	{
		if (ipiv[k] > 0)  // s=1
		{
			D[k + k*lda] = A[k + k*lda];
			for (int i=k+1; i<m; i++)
				L[i + k*lda] = A[i + k*lda];
		}
		else   // s=2
		{
			D[k + k*lda] = A[k + k*lda];
			D[(k+1) + (k+1)*lda] = A[(k+1) + (k+1)*lda];
			D[k + (k+1)*lda] = D[(k+1) + k*lda] = A[(k+1) + k*lda];

			for (int i=k+2; i<m; i++)
			{
				L[i + k*lda] = A[i + k*lda];
				L[i + (k+1)*lda] = A[i + (k+1)*lda];
			}
			k++;
		}
	}


	Show_mat(m,m,OA);
	Show_mat(m,m,D);
	Show_mat(m,m,L);

	double* W = new double[m*m];
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
			m, m, m, 1.0, L, lda, D, m, 0.0, W, m);
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
			m, m, m, -1.0, W, lda, L, m, 1.0, OA, m);

	delete [] W;

	cout << "Debug mode: \n";
	cout << "|| A - L*D*L^T ||_2 = " << cblas_dnrm2(m*m, OA, 1) << endl;

	delete [] OA;
	delete [] D;
	delete [] L;
	#endif
	////////// Debug mode //////////

	delete [] A;
	delete [] ipiv;

	return EXIT_SUCCESS;
}

