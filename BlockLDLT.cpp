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
    srand(20200409);
    // srand(time(NULL));

    // #pragma omp parallel for
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
    for (int i=0; i<m; i++)
    {
        for (int j=0; j<n; j++)
            printf("% 6.4lf, ",A[i + j*m]);
        cout << endl;
    }
    cout << endl;
}

// Show tile matrix
void Show_tilemat(const int m, const int n, const int mb, const int nb, double* A)
{
    const int p =  (m % mb == 0) ? m/mb : m/mb+1;   // # tile rows
    const int q =  (n % nb == 0) ? n/nb : n/nb+1;   // # tile columns

    for (int i=0; i<m; i++)
    {
        int ib = min(m-(i/mb)*mb,mb);
        for (int j=0; j<n; j++)
        {
            int ii = i/mb;
            int jj = j/nb;
            int ti = i - ii*mb;
            int tj = j - jj*nb;
            
            printf("% 6.4lf, ",A[ii*(mb*nb) + jj*(m*nb) + ti + tj*ib]);
        }
        cout << endl;
    }
    cout << endl;
}

void cm2ccrb(const int m, const int n, const int mb, const int nb, double* A, double* B)
{
    const int p =  (m % mb == 0) ? m/mb : m/mb+1;   // # tile rows
    const int q =  (n % nb == 0) ? n/nb : n/nb+1;   // # tile columns

    #pragma omp parallel for
    for (int j=0; j<q; j++)
    {
        int jb = min(n-j*nb,nb);

        for (int i=0; i<p; i++)
        {
            int ib = min(m-i*mb,mb);
            double* Aij = A+((i*mb)+(j*nb)*m);
            double* Bij = B+((i*mb*nb)+(j*nb*m));

            for (int jj=0; jj<jb; jj++)
                for (int ii=0; ii<ib; ii++)
                    Bij[ ii+jj*ib ] = Aij[ ii+jj*m ];
        }
    }
}

void ccrb2cm(const int m, const int n, const int mb, const int nb, double* B, double* A)
{
    const int p =  (m % mb == 0) ? m/mb : m/mb+1;   // # tile rows
    const int q =  (n % nb == 0) ? n/nb : n/nb+1;   // # tile columns

    #pragma omp parallel for
    for (int j=0; j<q; j++)
    {
        int jb = min(m-j*nb,nb);

        for (int i=0; i<p; i++)
        {
            int ib = min(m-i*mb,mb);
            double* Aij = A+((i*mb)+(j*nb)*m);
            double* Bij = B+((i*mb*nb)+(j*nb*m));

            for (int jj=0; jj<jb; jj++)
                for (int ii=0; ii<ib; ii++)
                    Aij[ ii+jj*m ] = Bij[ ii+jj*ib ];
        }
    }
}

// Serial LDLT factorization
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
// #define DEBUG

// Trace mode
#define TRACE

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

    const int m = atoi(argv[1]);       // # rows and columns <- square matrix
    const int b = atoi(argv[2]);       // tile size
    const int p =  (m % b == 0) ? m/b : m/b+1;   // # tiles

    double* A = new double [m*m];      // Original matrix
    double* B = new double [m*m];      // Tiled matrix
    const int lda = m;                 // Leading dimension of A

    double* DD = new double [m];       // DD_k = Diagonal elements of D_{kk}
    double* WD = new double [b*m];     // WD_k = L_{kk}*D_{kk}
    double* LD = new double [b*m];     // LD_k = L_{ik}*D_{kk}
    const int ldd = b;                 // Leading dimension of LD and WD

    for (int i=0; i<b*m; i++)          // Initialize WD to zero
        WD[i] = 0.0;
    /////////////////////////////////////////////////////////

    Gen_rand_lower_mat(m,m,A);         // Randomize elements of orig. matrix

    /////////////////////////////////////////////////////////
    #ifdef DEBUG
    double *OA = new double[m*m];    // OA: copy of A
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
    /////////////////////////////////////////////////////////

    double timer = omp_get_wtime();   // Timer start

    /////////////////////////////////////////////////////////
    #pragma omp parallel
    {
        #pragma omp single
        {
            // Convert CM to CCRB
            for (int j=0; j<p; j++)
            {
                int jb = min(m-j*b,b);
                for (int i=j; i<p; i++)
                {
                    int ib = min(m-i*b,b);
                    double* Aij = A+((i*b)+(j*b)*m);
                    double* Bij = B+((i*b*b)+(j*b*m));

                    #pragma omp task depend(in: Aij[0:m*jb]) depend(out: Bij[0:ib*jb]) priority(max(2,p-min(i,j)-1))
                    {
                        #ifdef TRACE
                        trace_cpu_start();
                        trace_label("Yellow", "Conv.");
                        #endif

                        for (int jj=0; jj<jb; jj++)
                            for (int ii=0; ii<ib; ii++)
                                Bij[ ii+jj*ib ] = Aij[ ii+jj*m ];

                        #ifdef TRACE
                        trace_cpu_stop("Yellow");
                        #endif
                    }
                }
            }

            for (int k=0; k<p; k++)
            {
                int kb = min(m-k*b,b);
                double* Bkk = B+((k*b*b)+(k*b)*lda); // Bkk: Top address of B_{kk}
                double* Dk = DD+k*ldd;               // Dk: diagnal elements of D_{kk}
                double* Wkk = WD+(k*ldd*ldd);        // Wkk: Top address of L_{kk} * D_{kk}

                #pragma omp task \
                    depend(inout: Bkk[0:kb*kb]) \
                    depend(out: DD[k*ldd:kb], WD[k*ldd*ldd:kb*kb]) priority(p)
                {
                    #ifdef TRACE
                    trace_cpu_start();
                    trace_label("Red", "DSYTRF");
                    #endif

                    dsytrf(kb,kb,Bkk);          // DSYTRF

                    for (int i=0; i<kb; i++)    // Set Dk
                        Dk[i] = Bkk[i+i*kb];

                    for (int j=0; j<kb; j++)    // Set Wkk
                        for (int i=j; i<kb; i++)
                            Wkk[i+j*ldd] = (i==j) ? Dk[j] : Dk[j]*Bkk[i+j*kb];

                    #ifdef TRACE
                    trace_cpu_stop("Red");
                    #endif
                }

                for (int i=k+1; i<p; i++)
                {
                    int ib = min(m-i*b,b);
                    double* Bik = B+((i*b*b)+(k*b)*lda); // Bik: Top address of B_{ik}
                    double* LDk = LD+(k*ldd*ldd);        // LDk:

                    #pragma omp task \
                        depend(in: DD[k*ldd:kb], WD[k*ldd*ldd:kb*kb]) \
                        depend(inout: Bik[0:ib*kb]) \
                        depend(out: LD[k*ldd*ldd:kb*kb]) priority(max(5,p-i-2))
                    {
                        #ifdef TRACE
                        {
                            trace_cpu_start();
                            trace_label("Green", "DTRSM");
                        }
                        #endif

                        // Updatre B_{ik}
                        cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                                    ib, kb, 1.0, Wkk, ldd, Bik, ib);

                        for (int l=0; l<kb; l++)       // LD_k = L_{ik}*D_{kk}
                        {
                            cblas_dcopy(ib, Bik+l*ib, 1, LDk+l*ldd, 1);
                            cblas_dscal(ib, DD[l+k*ldd], LDk+l*ldd, 1);
                        }

                        #ifdef TRACE
                        {
                            trace_cpu_stop("Green");
                        }
                        #endif
                    }

                    for (int j=k+1; j<=i; j++)
                    {
                        int jb = min(m-j*b,b);
                        double *Bij = B+((i*b*b)+(j*b)*lda);
                        double *Ljk = B+((j*b*b)+(k*b)*lda);

                        #pragma omp task \
                            depend(in: LD[k*ldd*ldd:kb*kb], Ljk[0:jb*kb]) \
                            depend(inout: Bij[0:ib*jb])priority(max(5,p-min(i,j)-3))
                        {
                            #ifdef TRACE
                            {
                                if (i==j) {
                                    trace_cpu_start();
                                    trace_label("Cyan", "DSYDRK");
                                } else {
                                    trace_cpu_start();
                                    trace_label("Blue", "DGEMDM");
                                }
                            }
                            #endif

                            // Update B_{ij}
                            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                        ib, jb, kb, -1.0, LDk, ldd, Ljk, jb, 1.0, Bij, ib);

                            // Banish upper part of A_{ii}
                            if (i==j)
                                for (int ii=0; ii<ib; ii++)
                                    for (int jj=ii+1; jj<jb; jj++)
                                        Bij[ii+jj*ib] = 0.0;

                            #ifdef TRACE
                            {
                                if (i==j)
                                    trace_cpu_stop("Cyan");
                                else
                                    trace_cpu_stop("Blue");
                            }
                            #endif
                        }
                    }
                } // End of i-loop
            } // End of k-loop

            // Convert CCRB to CM
            for (int j=0; j<p; j++)
            {
                int jb = min(m-j*b,b);
                for (int i=j; i<p; i++)
                {
                    int ib = min(m-i*b,b);
                    double* Aij = A+((i*b)+(j*b)*m);
                    double* Bij = B+((i*b*b)+(j*b*m));

                    #pragma omp task depend(in: Bij[0:ib*jb]) depend(out: Aij[0:m*jb])
                    {
                        #ifdef TRACE
                        trace_cpu_start();
                        trace_label("Violet", "Conv.");
                        #endif

                        for (int jj=0; jj<jb; jj++)
                            for (int ii=0; ii<ib; ii++)
                                Aij[ ii+jj*m ] = Bij[ ii+jj*ib ];
                        
                        #ifdef TRACE
                        trace_cpu_stop("Violet");
                        #endif
                    }
                }
            }
        } // End of single region
    } // End of parallel region
    /////////////////////////////////////////////////////////

    timer = omp_get_wtime() - timer; // Timer stop
    cout << m << ", " << timer << endl;

    /////////////////////////////////////////////////////////
    #ifdef DEBUG
    // Make L and D
    for (int k=0; k<m; k++)
    {
        D[k + k*lda] = A[k + k*lda];
        for (int i=k+1; i<m; i++)
            L[i + k*lda] = A[i + k*lda];
    }

    double* W = new double[m*m];
    // W <- L*D
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, m, m, 1.0, L, lda, D, m, 0.0, W, m);
    // OA <- W*L^T - OA
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                m, m, m, -1.0, W, lda, L, m, 1.0, OA, m);
    delete [] W;

    cout << "Debug mode: \n";
    cout << "|| A - L*D*L^T ||_2 = " << cblas_dnrm2(m*m, OA, 1) << endl;

    delete [] OA;
    delete [] D;
    delete [] L;
    #endif
    /////////////////////////////////////////////////////////

    delete [] A;
    delete [] B;
    delete [] DD;
    delete [] WD;
    delete [] LD;

    return EXIT_SUCCESS;
}

