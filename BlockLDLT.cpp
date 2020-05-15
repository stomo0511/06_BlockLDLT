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
	double* OA = new double [m*m];   // Copy of original matrix
	cblas_dcopy(m*m, A, 1, OA, 1);
	#endif
	////////// Debug mode //////////

	double timer = omp_get_wtime();    // Timer start

	assert(0 == LAPACKE_dsytrf(MKL_COL_MAJOR, 'L', m, A, lda, ipiv));

	timer = omp_get_wtime() - timer;   // Timer stop

	cout << m << ", " << timer << endl;

	////////// Debug mode //////////
	#ifdef DEBUG
	cout << "Debug mode: \n";

	double* b = new double [m];      // RHS vector
	double* x = new double [m];      // Solution vector
	for (int i=0; i<m; i++)
		b[i] = x[i] = 1.0;

	assert(0 == LAPACKE_dsytrs(LAPACK_COL_MAJOR, 'L', m, 1, A, lda, ipiv, x, lda));

	cblas_dsymv(CblasColMajor, CblasLower, m, -1.0, OA, lda, x, 1, 1.0, b, 1);

	cout << "|| b - A*x ||_2 = " << cblas_dnrm2(m, b, 1) << endl;

	delete [] OA;
	delete [] b;
	delete [] x;
	#endif
	////////// Debug mode //////////

	delete [] A;
	delete [] ipiv;

	return EXIT_SUCCESS;
}

