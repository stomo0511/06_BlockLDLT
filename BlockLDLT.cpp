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
void Gen_rand_lower_mat(const int m, const int n, double* A)
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
void Show_mat(const int m, const int n, double* A)
{
	for (int i=0; i<m; i++) {
		for (int j=0; j<n; j++)
			printf("% 6.4lf, ",A[i + j*m]);
		cout << endl;
	}
	cout << endl;
}

void dsytrf(const int m, const int lda, double* A)
{
	double* v = new double [m];
	for (int k=0; k<m; k++)
	{
		for (int i=0; i<k; i++)
			v[i] = A[k+i*lda]*A[i+i*lda];

		v[k] = A[k+k*lda] - cblas_ddot(k,A+k,lda,v,1);
		A[k+k*lda] = v[k];

		cblas_dgemv(CblasColMajor, CblasNoTrans,
				m-k-1, k, -1.0, A+(k+1), lda, v, 1, 1.0, A+(k+1)+k*lda,1);
		cblas_dscal(m-k-1, 1.0/v[k], A+(k+1)+k*lda, 1);
	}
	delete [] v;
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
	// Usage "a.out [size of matrix: m ] [tile size: b]"
	assert(argc > 2);

	const int m = atoi(argv[1]);     // # rows and columns <- square matrix
	const int b = atoi(argv[2]);     // tile size
	const int p =  (m % b == 0) ? m/b : m/b+1;   // # tiles

	double* A = new double [m*m];    // Original matrix
	const int lda = m;               // Leading dimension of A
	double* d = new double [b];      // Diagonal elements of D_{kk}
	double* LD = new double [b*b];   // L_{ik}*D_{kk}
	const int ldd = b;               // Leading dimension of LD

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

//	Show_mat(m,m,A);

	double timer = omp_get_wtime();    // Timer start

	for (int k=0; k<p; k++)
	{
		int kb = min(m-k*b,b);
		double *Akk = A+((k*b)+(k*b)*lda);

		// DSYTRF
		dsytrf(kb,lda,Akk);

		for (int i=0; i<kb; i++)    // d: diagnal elements of D_{kk}
			d[i] = Akk[i+i*lda];

		for (int i=k+1; i<p; i++)
		{
			int ib = min(m-i*b,b);

			// Temprarily transform A_{kk} <- L_{kk} * D_{kk}
			for (int l=0; l<kb-1; l++)
				cblas_dscal(kb-(l+1), d[l], Akk+(l+1)+l*lda, 1);

			double *Aik = A+((i*b)+(k*b)*lda);

			// Generate L_{ik}
			cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
						ib, kb, 1.0, Akk, lda, Aik, lda);

			// Restore A_{kk}
			for (int l=0; l<kb-1; l++)
				cblas_dscal(kb-(l+1), 1.0/d[l], Akk+(l+1)+l*lda, 1);

			// LD = L_{ik}*D_{kk}
			for (int l=0; l<kb; l++)
			{
				cblas_dcopy(ib, Aik+l*lda, 1, LD+l*ldd, 1);
				cblas_dscal(ib, d[l], LD+l*ldd, 1);
			}

			for (int j=k+1; j<=i; j++)
			{
				int jb = min(m-j*b,b);
				double *Aij = A+((i*b)+(j*b)*lda);
				double *Ljk = A+((j*b)+(k*b)*lda);
				cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
						ib, jb, kb, -1.0, LD, ldd, Ljk, lda, 1.0, Aij, lda);

				// Banish upper part of A_{ii}
				if (i==j)
					for (int ii=0; ii<ib; ii++)
						for (int jj=ii+1; jj<jb; jj++)
							Aij[ii+jj*lda] = 0.0;
			}
		}
	}

	timer = omp_get_wtime() - timer;   // Timer stop

	cout << "m = " << m << ", time = " << timer << endl;

	Show_mat(m,m,A);

	////////// Debug mode //////////
	#ifdef DEBUG
	// Make L and D
	for (int k=0; k<m; k++)
	{
		D[k + k*lda] = A[k + k*lda];
		for (int i=k+1; i<m; i++)
			L[i + k*lda] = A[i + k*lda];
	}

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
	delete [] d;
	delete [] LD;

	return EXIT_SUCCESS;
}

