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
    // srand(20200409);
    srand(time(NULL));

    // #pragma omp parallel for
    for (int j = 0; j < n; j++)
        for (int i = j; i < m; i++)
            A[i + j * m] = 10.0 - 20.0 * (double)rand() / RAND_MAX;
}

// Show matrix
void Show_mat(const int m, const int n, double *A)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
            printf("% 6.4lf, ", A[i + j * m]);
        cout << endl;
    }
    cout << endl;
}

// Show tile matrix
void Show_tilemat(const int m, const int n, const int mb, const int nb, double *A)
{
    const int p = (m % mb == 0) ? m / mb : m / mb + 1; // # tile rows
    const int q = (n % nb == 0) ? n / nb : n / nb + 1; // # tile columns

    for (int i = 0; i < m; i++)
    {
        int ii = i / mb;
        int ib = min(m - ii * mb, mb);
        int ti = i - ii * mb;
        for (int j = 0; j < n; j++)
        {
            int jj = j / nb;
            int jb = min(n - jj * nb, nb);
            int tj = j - jj * nb;

            printf("% 6.4lf, ", A[m * jj * nb + ii * mb * jb + tj * ib + ti]);
        }
        cout << endl;
    }
    cout << endl;
}

void cm2ccrb(const int m, const int n, const int mb, const int nb, const double *A, double *B)
{
    const int p = (m % mb == 0) ? m / mb : m / mb + 1; // # tile rows
    const int q = (n % nb == 0) ? n / nb : n / nb + 1; // # tile columns

    for (int j = 0; j < q; j++)
    {
        int jb = min(n - j * nb, nb);
        for (int i = j; i < p; i++)
        {
            int ib = min(m - i * mb, mb);
            const double *Aij = A + (j * nb * m + i * mb);
            double *Bij = B + (j * nb * m + i * mb * jb);

#pragma omp task depend(in                           \
                        : Aij [0:m * jb]) depend(out \
                                                 : Bij [0:ib * jb])
            {
#ifdef TRACE
                trace_cpu_start();
                trace_label("Yellow", "Conv.");
#endif

                for (int jj = 0; jj < jb; jj++)
                    for (int ii = 0; ii < ib; ii++)
                        Bij[ii + jj * ib] = Aij[ii + jj * m];

#ifdef TRACE
                trace_cpu_stop("Yellow");
#endif
            }
        }
    }
}

void ccrb2cm(const int m, const int n, const int mb, const int nb, const double *B, double *A)
{
    const int p = (m % mb == 0) ? m / mb : m / mb + 1; // # tile rows
    const int q = (n % nb == 0) ? n / nb : n / nb + 1; // # tile columns

    for (int j = 0; j < q; j++)
    {
        int jb = min(m - j * nb, nb);
        for (int i = j; i < p; i++)
        {
            int ib = min(m - i * mb, mb);
            double *Aij = A + (j * nb * m + i * nb);
            const double *Bij = B + (j * nb * m + i * nb * jb);

#pragma omp task depend(in                            \
                        : Bij [0:ib * jb]) depend(out \
                                                  : Aij [0:m * jb])
            {
#ifdef TRACE
                trace_cpu_start();
                trace_label("Violet", "Conv.");
#endif

                for (int jj = 0; jj < jb; jj++)
                    for (int ii = 0; ii < ib; ii++)
                        Aij[ii + jj * m] = Bij[ii + jj * ib];

#ifdef TRACE
                trace_cpu_stop("Violet");
#endif
            }
        }
    }
}

// Serial LDLT factorization
void dsytrf(const int m, const int lda, double *A)
{
    double *v = new double[m];
    for (int k = 0; k < m; k++)
    {
        for (int i = 0; i < k; i++)
            v[i] = A[k + i * lda] * A[i + i * lda];

        v[k] = A[k + k * lda] - cblas_ddot(k, A + k, lda, v, 1);
        A[k + k * lda] = v[k];

        cblas_dgemv(CblasColMajor, CblasNoTrans,
                    m - k - 1, k, -1.0, A + (k + 1), lda, v, 1, 1.0, A + (k + 1) + k * lda, 1);
        cblas_dscal(m - k - 1, 1.0 / v[k], A + (k + 1) + k * lda, 1);
    }
    delete[] v;
}

// Trace mode
// #define TRACE

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
        cerr << "usage: a.out[size of matrix: m ] [tile size: b]\n";
        return EXIT_FAILURE;
    }

    const int m = atoi(argv[1]);                       // # rows and columns <- square matrix
    const int nb = atoi(argv[2]);                      // tile size
    const int p = (m % nb == 0) ? m / nb : m / nb + 1; // # tiles

    double *A = new double[m * m];  // Original matrix
    double *OA = new double[m * m]; // OA: copy of A
    double *B = new double[m * m];  // Tiled matrix
    const int lda = m;              // Leading dimension of A

    double *DD = new double[m];      // DD_k = Diagonal elements of D_{kk}
    double *LD = new double[nb * m]; // LD_k = L_{ik}*D_{kk}
    const int ldd = nb;              // Leading dimension of LD
    /////////////////////////////////////////////////////////

    Gen_rand_lower_mat(m, m, A); // Randomize elements of orig. matrix
    cblas_dcopy(m * m, A, 1, OA, 1);

    double timer = omp_get_wtime(); // Timer start

    /////////////////////////////////////////////////////////
    cm2ccrb(m, m, nb, nb, A, B); // Convert CM(A) to CCRB(B)

    // Blocked LDLT part start
    for (int k = 0; k < p; k++)
    {
        int kb = min(m - k * nb, nb);
        double *Bkk = B + (k * nb * lda + k * nb * kb); // Bkk: Top address of B_{kk}
        double *Dk = DD + k * ldd;                      // Dk: diagnal elements of D_{kk}

        ///////////////////////////////
        // DSYTRF: B_{kk} -> L_{kk}, D_{kk}
        dsytrf(kb, kb, Bkk); // DSYTRF

        for (int i = 0; i < kb; i++) // Set Dk
            Dk[i] = Bkk[i + i * kb];


        for (int i = k + 1; i < p; i++)
        {
            int ib = min(m - i * nb, nb);
            double *Bik = B + (k * nb * lda + i * nb * kb); // Bik: Top address of B_{ik}
            double *LDk = LD + (k * ldd * ldd);             // LDk:

            ///////////////////////////////
            // TRSM: B_{ik} -> L_{ik}
            cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasUnit,
                        ib, kb, 1.0, Bkk, kb, Bik, ib);

            for (int l = 0; l < kb; l++)
            {
                cblas_dscal(ib, 1.0 / Dk[l], Bik + l * ib, 1);      // B_{ik} <- B_{ik} D_{kk}^{-1}
                cblas_dcopy(ib, Bik + l * ib, 1, LDk + l * ldd, 1); // LD_k = L_{ik}*D_{kk}
                cblas_dscal(ib, Dk[l], LDk + l * ldd, 1);
            }

            for (int j = k + 1; j <= i; j++)
            {
                int jb = min(m - j * nb, nb);
                double *Bij = B + (j * nb * lda + i * nb * jb);
                double *Ljk = B + (k * nb * lda + j * nb * kb);

                // Update B_{ij}
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                            ib, jb, kb, -1.0, LDk, ldd, Ljk, jb, 1.0, Bij, ib);

                // Banish upper part of A_{ii}
                if (i == j)
                    for (int ii = 0; ii < ib; ii++)
                        for (int jj = ii + 1; jj < jb; jj++)
                            Bij[ii + jj * ib] = 0.0;

            }
        } // End of i-loop
    } // End of k-loop
    ccrb2cm(m, m, nb, nb, B, A); // Convert CCRB(B) to CM(A)
    /////////////////////////////////////////////////////////

    timer = omp_get_wtime() - timer; // Timer stop
    cout << m << ", " << timer << ", ";

    /////////////////////////////////////////////////////////
    double *b = new double[m]; // RHS vector
    double *x = new double[m]; // Solution vector
    for (int i = 0; i < m; i++)
        b[i] = x[i] = 1.0;

    timer = omp_get_wtime(); // Timer start

    // Solve L*x = b for x
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                m, 1, 1.0, A, lda, x, lda);

    // x := D^{-1} x
    for (int i = 0; i < m; i++)
        x[i] /= A[i + i * lda];

    // Solbe L^{T}*y = x for y(x)
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasUnit,
                m, 1, 1.0, A, lda, x, lda);

    timer = omp_get_wtime() - timer; // Timer stop
    // cout << m << ", " << timer << endl;
    cout << timer << ", ";

    // b := b - A*x
    cblas_dsymv(CblasColMajor, CblasLower, m, -1.0, OA, lda, x, 1, 1.0, b, 1);
    // cout << "No piv LDLT:    || b - A*x ||_2 = " << cblas_dnrm2(m, b, 1) << endl;
    cout << cblas_dnrm2(m, b, 1) << ", ";

    ////////// Iterative refinement //////////
    double *r = new double[m]; // Residure vector
    cblas_dcopy(m, b, 1, r, 1);
    for (int i = 0; i < m; i++)
        b[i] = 1.0;

    timer = omp_get_wtime(); // Timer start

    // Solve L*y = r for y(r)
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                m, 1, 1.0, A, lda, r, lda);

    // r := D^{-1} r
    for (int i = 0; i < m; i++)
        r[i] /= A[i + i * lda];

    // Solbe L^{T}*y = r for y(r)
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasUnit,
                m, 1, 1.0, A, lda, r, lda);

    // x := x+r
    cblas_daxpy(m, 1.0, r, 1, x, 1);

    timer = omp_get_wtime() - timer; // Timer stop
    // cout << m << ", " << timer << endl;
    cout << timer << ", ";

    // b := b - A*x
    cblas_dsymv(CblasColMajor, CblasLower, m, -1.0, OA, lda, x, 1, 1.0, b, 1);
    // cout << "Apply 1 it ref: || b - A*x ||_2 = " << cblas_dnrm2(m, b, 1) << endl;
    cout << cblas_dnrm2(m, b, 1) << endl;

    /////////////////////////////////////////////////////////

    delete[] A;
    delete[] OA;
    delete[] B;
    delete[] DD;
    delete[] LD;
    delete[] r;
    delete[] b;
    delete[] x;

    return EXIT_SUCCESS;
}
