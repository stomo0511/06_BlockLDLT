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
	srand(time(NULL));
	// srand(20200314);

//	#pragma omp parallel for
	for (int j=0; j<n; j++)
		for (int i=j; i<m; i++)
			A[i+j*m] = 1.0 - 2*(double)rand() / RAND_MAX;
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

// Apply iterative refinement
#define ITREF

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
	if (argc < 3)
	{
		cerr << "Usage a.out [size of matrix: m ] [tile size: b]\n";
		return EXIT_FAILURE;
	}

	const int m = atoi(argv[1]);     // # rows and columns <- square matrix
	const int nb = atoi(argv[2]);    // tile size
	const int p =  (m % nb == 0) ? m/nb : m/nb+1;   // # tiles

	double* A = new double [m*m];    // Original matrix
	double* DD = new double [m];     // DD_k = Diagonal elements of D_{kk}
	double* WD = new double [nb*m];  // WD_k = L_{kk}*D_{kk}
	double* LD = new double [nb*m];  // LD_k = L_{ik}*D_{kk}
	const int lda = m;               // Leading dimension of A
	const int ldd = nb;              // Leading dimension of LD

	int* P = new int [p*p];          // Progress table
	const int ldp = p;

	for (int i=0; i<nb*m; i++)       // Zero initialize of WD
		WD[i] = 0.0;

	Gen_rand_lower_mat(m,m,A);       // Randomize elements of orig. matrix

	////////// Debug mode //////////
	#ifdef DEBUG
	double *OA = new double[m*m];
	cblas_dcopy(m*m, A, 1, OA, 1);
	for (int i=0; i<m; i++)
		for (int j=0; j<=i; j++)
			OA[j+i*m] = OA[i+j*m];   // Fill the upper triangular part
	#endif
	////////// Debug mode //////////

	double timer = omp_get_wtime();    // Timer start

	/////////////////////////////////////////////////////////
	#pragma omp parallel
	{
		#pragma omp single
		{
			for (int k=0; k<p; k++)
			{
				int kb = min(m-k*nb,nb);
				double* Akk = A+((k*nb)+(k*nb)*lda);   // Akk: Top address of A_{kk}
				double* Dk = DD+(k*ldd);               // Dk: diagnal elements of D_{kk}
				double* Wkk = WD+(k*ldd*ldd);          // Wkk: Top address of L_{kk} * D_{kk}

				#pragma omp task \
					depend(inout: P[k+k*ldp]) \
					depend(out: DD[k*ldd:kb], WD[k*ldd*ldd:kb*kb])
				{
					#ifdef TRACE
					trace_cpu_start();
					trace_label("Red", "DSYTRF");
					#endif

					dsytrf(kb,lda,Akk);         // DSYTRF

					for (int l=0; l<kb; l++)    // Set Dk
						Dk[l] = Akk[l+l*lda];

					for (int j=0; j<kb; j++)    // Set Wkk
						for (int i=j; i<kb; i++)
							Wkk[i+j*ldd] = (i==j) ? Dk[j] : Dk[j]*Akk[i+j*lda];

					#ifdef TRACE
					trace_cpu_stop("Red");
					#endif
				}

				for (int i=k+1; i<p; i++)
				{
					int ib = min(m-i*nb,nb);
					double* Aik = A+((i*nb)+(k*nb)*lda);  // Aik: Top address of A_{ik}
					double* LDk = LD+(k*ldd*ldd);         // LDk:

					#pragma omp task \
						depend(in: DD[k*ldd:kb], WD[k*ldd*ldd:kb*kb]) \
						depend(inout: P[i+k*ldp]) \
						depend(out: LD[k*ldd*ldd:kb*kb])
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
						int jb = min(m-j*nb,nb);
						double *Aij = A+((i*nb)+(j*nb)*lda);
						double *Ljk = A+((j*nb)+(k*nb)*lda);

						#pragma omp task \
							depend(in: LD[k*ldd*ldd:kb*kb], P[j+k*ldp]) \
							depend(inout: P[i+j*ldp])
						{
							#ifdef TRACE
							if (i==j) {
								trace_cpu_start();
								trace_label("Cyan", "DSYDRK");
							} else {
								trace_cpu_start();
								trace_label("Blue", "DGEMDM");
							}
							#endif

							// Update A_{ij}
							cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
									ib, jb, kb, -1.0, LDk, ldd, Ljk, lda, 1.0, Aij, lda);

							// Banish upper part of A_{ii}
							if (i==j)
								for (int ii=0; ii<ib; ii++)
									for (int jj=ii+1; jj<jb; jj++)
										Aij[ii+jj*lda] = 0.0;

							#ifdef TRACE
							if (i==j)
								trace_cpu_stop("Cyan");
							else
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

	// cout << m << ", " << timer << endl;
	cout << m << ", " << timer << ", ";

	////////// Debug mode //////////
	#ifdef DEBUG
	// cout << "Debug mode: \n";

	double* b = new double [m];        // RHS vector
	double* x = new double [m];        // Solution vector
	for (int i=0; i<m; i++)
		b[i] = x[i] = 1.0;
	
	timer = omp_get_wtime();    // Timer start

	// Solve L*x = b for x
	cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
		m, 1, 1.0, A, lda, x, lda);

	// x := D^{-1} x
	for (int i=0; i<m; i++)
		x[i] /= DD[i];
	
	// Solbe L^{T}*y = x for y(x)
	cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasUnit,
		m, 1, 1.0, A, lda, x, lda);

	timer = omp_get_wtime() - timer;   // Timer stop
	// cout << m << ", " << timer << endl;
	cout << timer << ", ";

	// b := b - A*x
	cblas_dsymv(CblasColMajor, CblasLower, m, -1.0, OA, lda, x, 1, 1.0, b, 1);
	// cout << "No piv LDLT:    || A - L*D*L^T ||_2 = " << cblas_dnrm2(m, b, 1) << endl;
	cout << cblas_dnrm2(m, b, 1) << ", ";

	////////// Iterative refinement //////////
	#ifdef ITREF
	double* r = new double [m];       // Residure vector
	cblas_dcopy(m,b,1,r,1);
	for (int i=0; i<m; i++)
		b[i] = 1.0;

	timer = omp_get_wtime();    // Timer start

	// Solve L*y = r for y(r)
	cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
		m, 1, 1.0, A, lda, r, lda);

	// r := D^{-1} r
	for (int i=0; i<m; i++)
		r[i] /= DD[i];
	
	// Solbe L^{T}*y = r for y(r)
	cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasUnit,
		m, 1, 1.0, A, lda, r, lda);

	// x := x+r
	cblas_daxpy(m,1.0,r,1,x,1);

	timer = omp_get_wtime() - timer;   // Timer stop
	// cout << m << ", " << timer << endl;
	cout << timer << ", ";

	// b := b - A*x
	cblas_dsymv(CblasColMajor, CblasLower, m, -1.0, OA, lda, x, 1, 1.0, b, 1);
	// cout << "Apply 1 it ref: || A - L*D*L^T ||_2 = " << cblas_dnrm2(m, b, 1) << endl;
	cout << cblas_dnrm2(m, b, 1) << endl;

	delete [] r;
	#endif
	////////// Iterative refinement //////////

	delete [] OA;
	delete [] b;
	delete [] x;
	#endif
	////////// Debug mode //////////

	delete [] A;
	delete [] DD;
	delete [] WD;
	delete [] LD;
	delete [] P;

	return EXIT_SUCCESS;
}

