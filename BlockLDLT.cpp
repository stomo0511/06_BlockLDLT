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
	srand(time(NULL));
//	srand(20200314);

    #pragma omp parallel for
	for (int j=0; j<n; j++)
		for (int i=j; i<m; i++)
				A[i+j*m] = 10.0 - 20.0*(double)rand() / RAND_MAX;
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

#define MAX_LC 10

int main(const int argc, const char **argv)
{
	// Usage "a.out [size of matrix: m ]"
	if (argc < 2)
	{
		cerr << "Usage: a.out [size of matrix]" << endl;
		return EXIT_FAILURE;
	}

	const int m = atoi(argv[1]);     // # rows and columns <- the matrix is square
	const int lda = m;

	double* A = new double [m*m];    // Original matrix
	double* OA = new double [m*m];   // Copy of original matrix
	int* ipiv = new int[m];          // Pivot vector

	double* b = new double [m];      // RHS vector
	double* x = new double [m];      // Solution vector
    double* r = new double [m];      // residure vector

	Gen_rand_lower_mat(m,m,OA);      // Randomize elements of orig. matrix

	for (int lc = 0; lc < MAX_LC; lc++)
	{
		cblas_dcopy(m*m, OA, 1, A, 1);

		for (int i=0; i<m; i++)
			x[i] = b[i] = (double)(1.0);

		double timer = omp_get_wtime();    // Timer start
		assert(0 == LAPACKE_dsytrf(MKL_COL_MAJOR, 'L', m, A, lda, ipiv));
		timer = omp_get_wtime() - timer;   // Timer stop
		cout << m << ", " << timer << ", ";

		timer = omp_get_wtime();            // Timer start
		assert(0 == LAPACKE_dsytrs(LAPACK_COL_MAJOR, 'L', m, 1, A, lda, ipiv, x, lda));
		timer = omp_get_wtime() - timer;   // Timer stop
		cout << timer << ", ";

		///////////////////////////////////////////////////////////////////////////////
		// Check || A*x - b ||
		cblas_dsymv(CblasColMajor, CblasLower, m, -1.0, OA, lda, x, 1, 1.0, b, 1);
		cout << cblas_dnrm2(m, b, 1) << ", ";

		///////////////////////////////////////////////////////////////////////////////
		// Iterative refinement
		timer = omp_get_wtime();            // Timer start
		assert(0 == LAPACKE_dsytrs(LAPACK_COL_MAJOR, 'L', m, 1, A, lda, ipiv, b, lda));
		cblas_daxpy(m, 1.0, b, 1, x, 1);
		timer = omp_get_wtime() - timer;   // Timer stop
		cout << timer << ", ";

		///////////////////////////////////////////////////////////////////////////////
		// Check || A*x - b ||
		for (int i=0; i<m; i++)
			b[i] = (double)(1.0);

		cblas_dsymv(CblasColMajor, CblasLower, m, -1.0, OA, lda, x, 1, 1.0, b, 1);
		cout << cblas_dnrm2(m, b, 1) << endl;
	}

	delete [] A;
	delete [] OA;
	delete [] ipiv;
	delete [] b;
	delete [] x;
    delete [] r;

	return EXIT_SUCCESS;
}

