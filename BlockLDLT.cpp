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

//	#pragma omp parallel for
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
	double* DD = new double [m];     // DD_k = Diagonal elements of D_{kk}
	double* WD = new double [b*m];   // WD_k = L_{kk}*D_{kk}
	double* LD = new double [b*m];   // LD_k = L_{ik}*D_{kk}
	const int ldd = b;               // Leading dimension of LD

	int* P = new int [p*p];          // Progress table
	const int ldp = p;

	for (int i=0; i<b*m; i++)        // Zero initialize of WD
		WD[i] = 0.0;

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

	/////////////////////////////////////////////////////////
	#pragma omp parallel
	{
		#pragma omp single
		{
			for (int k=0; k<p; k++)
			{
				int kb = min(m-k*b,b);
				double* Akk = A+((k*b)+(k*b)*lda);   // Akk: Top address of A_{kk}
				double* Dk = DD+k*ldd;               // Dk: diagnal elements of D_{kk}
				double* Wkk = WD+(k*ldd);            // Wkk: Top address of L_{kk} * D_{kk}

				#pragma omp task \
					depend(inout: P[k+k*ldp]) \
					depend(out: DD[k*ldd:kb], WD[k*ldd:kb*kb])
				{
					#ifdef TRACE
					trace_cpu_start();
					trace_label("Red", "DSYTRF");
					#endif

					dsytrf(kb,lda,Akk);         // DSYTRF

					for (int i=0; i<kb; i++)    // Set Dk
						Dk[i] = Akk[i+i*lda];

					for (int j=0; j<kb; j++)    // Set Wkk
						for (int i=j; i<kb; i++)
							Wkk[i+j*ldd] = (i==j) ? Dk[j] : Dk[j]*Akk[i+j*lda];

					#ifdef TRACE
					trace_cpu_stop("Red");
					#endif
				}

				for (int i=k+1; i<p; i++)
				{
					int ib = min(m-i*b,b);
					double* Aik = A+((i*b)+(k*b)*lda);  // Aik: Top address of A_{ik}
					double* LDk = LD+(k*ldd);           // LDk:

					#pragma omp task \
						depend(in: DD[k*ldd:kb], WD[k*ldd:kb*kb]) \
						depend(inout: P[i+k*ldp]) \
						depend(out: LD[k*ldd:kb*kb])
					{
						#ifdef TRACE
						trace_cpu_start();
						trace_label("Green", "DTRSM");
						#endif

						// Updatre A_{ik}
						cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
									ib, kb, 1.0, Wkk, ldd, Aik, lda);

						for (int l=0; l<kb; l++)       // LD_k = L_{ik}*D_{kk}
						{
							cblas_dcopy(ib, Aik+l*lda, 1, LDk+l*ldd, 1);
							cblas_dscal(ib, DD[l+k*ldd], LDk+l*ldd, 1);
						}

						#ifdef TRACE
						trace_cpu_stop("Green");
						#endif
					}

					for (int j=k+1; j<=i; j++)
					{
						int jb = min(m-j*b,b);
						double *Aij = A+((i*b)+(j*b)*lda);
						double *Ljk = A+((j*b)+(k*b)*lda);

						#pragma omp task \
							depend(in: LD[k*lda:kb*kb], P[j+k*ldp]) \
							depend(inout: P[i+j*ldp])
						{
							#ifdef TRACE
							trace_cpu_start();
							trace_label("Blue", "DGEMM");
							#endif

							// Update A_{ij}
							cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
									ib, jb, kb, -1.0, LD+k*ldd, ldd, Ljk, lda, 1.0, Aij, lda);

							// Banish upper part of A_{ii}
							if (i==j)
								for (int ii=0; ii<ib; ii++)
									for (int jj=ii+1; jj<jb; jj++)
										Aij[ii+jj*lda] = 0.0;

							#ifdef TRACE
							trace_cpu_stop("Blue");
							#endif
						}
					}
				} // End of i-loop
			} // End of k-loop
		} // End of single region
	} // End of parallel region
	/////////////////////////////////////////////////////////

	timer = omp_get_wtime() - timer;   // Timer stop

	cout << m << ", " << timer << endl;

//	Show_mat(m,m,A);

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
	delete [] DD;
	delete [] WD;
	delete [] LD;
	delete [] P;

	return EXIT_SUCCESS;
}

