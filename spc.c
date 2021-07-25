#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <mpi.h>

#include "spc.h"

#define  NPROCS   576

#define  EPS    2.220446e-16

double  A[N][N];
double  b[M][N];
double  x[M][N];
double  c[N];


int     myid, numprocs;


void spc(double [N][N], double [M][N], double [M][N], int, int); 

void main(int argc, char* argv[]) {

     double  t0, t1, t2, t_w;
     double  dc_inv, d_mflops, dtemp, dtemp2, dtemp_t;

     int     ierr;
     int     i, j;
     int     ii;      
     int     ib;

     ierr = MPI_Init(&argc, &argv);
     ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
     ierr = MPI_Comm_size(MPI_COMM_WORLD, &numprocs);



      /* matrix generation --------------------------*/
      for(j=0; j<N; j++) {
        ii = 0;
        for(i=j; i<N; i++) {
          A[j][i] = (N-j) - ii;
          A[i][j] = A[j][i];
          ii++;
        }
      }
      /* end of matrix generation -------------------------- */

     /* set vector b  -------------------------- */
      for (i=0; i<N; i++) {
        b[0][i] = 0.0;
        for (j=0; j<N; j++) {
          b[0][i] += A[i][j];
        }
      }
      for (i=0; i<M; i++) {
        for (j=0; j<N; j++) {
          b[i][j] = b[0][j];
        }
      }
     /* ----------------------------------------------------- */


     /* Start of spc routine ----------------------------*/
     ierr = MPI_Barrier(MPI_COMM_WORLD);
     t1 = MPI_Wtime();

     spc(A, b, x, N, M);

     t2 = MPI_Wtime();
     t0 =  t2 - t1; 
     ierr = MPI_Reduce(&t0, &t_w, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
     /* End of spc routine --------------------------- */

     if (myid == 0) {

       printf("--------------------------- \n");
       printf("N = %d , M = %d \n",N,M);
       printf("LU solve time  = %lf [sec.] \n",t_w);

       d_mflops = 2.0/3.0*(double)N*(double)N*(double)N;
       d_mflops += 7.0/2.0*(double)N*(double)N;
       d_mflops += 4.0/3.0*(double)N*(double)M;
       d_mflops = d_mflops/t_w;
       d_mflops = d_mflops * 1.0e-6;
       printf(" %lf [MFLOPS] \n", d_mflops);

     }

     /* Verification routine ----------------- */
     ib = N / NPROCS;
     dtemp_t = 0.0;
     for(i=0; i<M; i++) {
       for(j=myid*ib; j<(myid+1)*ib; j++) {
         dtemp2 = x[i][j] - 1.0;
         dtemp_t += dtemp2*dtemp2;
       }
     }
     dtemp_t = sqrt(dtemp_t);
     /* -------------------------------------- */

     MPI_Reduce(&dtemp_t, &dtemp, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

     /* do not modify follows. -------- */ 
     if (myid == 0) {
       dtemp2 = (double)N*(double)N;      
       dtemp_t = EPS*(double)N*dtemp2*(double)M;
       dtemp_t = sqrt(dtemp_t);
       printf("Pass value: %e \n", dtemp_t);
       printf("Calculated value: %e \n", dtemp);
       if (dtemp > dtemp_t) {
          printf("Error! Test is falled. \n");
          exit(1);
       } 
       printf(" OK! Test is passed. \n");
       printf("--------------------------- \n");
     }
     /* ----------------------------------------- */


     ierr = MPI_Finalize();

     exit(0);
}


#include <arm_sve.h>

const int BWIDTH = 8;

// LU decomposition
void LU(double A[N][N], int n) {
  int i, j, k;
  int ib = (n + numprocs - 1) / numprocs;
  int i_start = myid * ib;
  int i_end = (myid + 1) * ib > n ? n : (myid + 1) * ib;
  double dtemp;
  double buf[n];

  for (k=0; k<n; k++) {
    int src_id = k / ib;
    if (myid == src_id) {
     dtemp = 1.0 / A[k][k];
     for (i=k+1; i<n; i++) {
        A[i][k] = A[i][k]*dtemp;   
        buf[i] = A[i][k];
     }
     for (int dst_id = myid + 1; dst_id < numprocs; ++dst_id) {
       MPI_Send(buf, n, MPI_DOUBLE, dst_id, k, MPI_COMM_WORLD);
     }
     ++i_start;
    } else if (src_id < myid) {
      MPI_Recv(buf, n, MPI_DOUBLE, src_id, k, MPI_COMM_WORLD, NULL);
    }
    for (j=k+1; j<n; j++) {
     //  dtemp = A[j][k];
      dtemp = buf[j];
      for (i=i_start; i<i_end; i++) {
        A[j][i] = A[j][i] - A[k][i]*dtemp; 
      }
    }
  }
}

// forward
void forward(double A[N][N], double b[M][N], double c[N][BWIDTH], int n, int m, int ne, int nw) {
  int i, j, k, l;
  double dtemp;

  int ib = (n + numprocs - 1) / numprocs;
  int i_start = myid * ib;
  int i_end = (myid + 1) * ib > n ? n : (myid + 1) * ib;

  for (i = 0; i < n; ++i) 
    for (l = 0; l < nw; ++l) c[i][l] = 0.0;

  for (i = i_start; i < n; i += ib) {
    if (myid != 0) {
      MPI_Recv(&c[i][0], ib*BWIDTH, MPI_DOUBLE, myid-1, i, MPI_COMM_WORLD, NULL);
    }
    if (myid == i / ib) {
      for (k = i; k < i + ib; ++k) {
        for (l = 0; l < nw; ++l) {
          c[k][l] = b[ne][k] + c[k][l];
          for (j = i_start; j < k; ++j) {
            c[k][l] -= A[k][j] * c[j][l];
          }
        }
      }
    } else {
      for (k = i; k < i + ib; ++k) {
        for (j = i_start; j < i_end; ++j) {
          // for (l = 0; l < nw; ++l) {
          //   c[k][l] -= A[k][j] * c[j][l];
          // }
          l = 0;
          svbool_t pg = svwhilelt_b64(l, nw);
          do {
            svfloat64_t csrc_vec = svld1(pg, &c[j][l]);
            svfloat64_t cdst_vec = svld1(pg, &c[k][l]);
            cdst_vec = svmls_n_f64_z(pg, cdst_vec, csrc_vec, A[k][j]);
            svst1(pg, &c[k][l], cdst_vec);
            l += svcntd();
            pg = svwhilelt_b64(l, nw);
          } while (svptest_any(svptrue_b64(), pg));
        }
      }
      if (myid != numprocs - 1) {
        MPI_Send(&c[i][0], ib*BWIDTH, MPI_DOUBLE, myid+1, i, MPI_COMM_WORLD);
      }
    }
  }
}

// backward
void backward(double A[N][N], double c[N][BWIDTH], double x[M][N], int n, int m, int ne, int nw) {
  int i, j, k, l;
  double dtemp;

  int ib = (n + numprocs - 1) / numprocs;
  int i_start = myid * ib;
  int i_end = (myid + 1) * ib > n ? n : (myid + 1) * ib;

  for (i = 0; i < n; ++i) x[ne][i] = 0.0;

  for (i = i_start; i >= 0; i -= ib) {
    if (myid != numprocs - 1) {
      for (l = 0; l < nw; ++l) {
        MPI_Recv(&x[ne+l][i], ib, MPI_DOUBLE, myid+1, i, MPI_COMM_WORLD, NULL);
      }
    }
    if (i == i_start) {
      for (l = 0; l < nw; ++l) {
        for (k = i + ib - 1; k >= i; --k) {
          x[ne+l][k] = c[k][l] + x[ne+l][k];
          for (j = i_end - 1; j > k; --j) {
            x[ne+l][k] -= A[k][j] * x[ne+l][j];
          }
          x[ne+l][k] = x[ne+l][k] / A[k][k];
        }
      }
    } else {
      for (l = 0; l < nw; ++l) {
        for (k = i + ib - 1; k >= i; --k) {
          for (j = i_end - 1; j >= i_start; --j) {
            x[ne+l][k] -= A[k][j] * x[ne+l][j];
          }
        }
      }
      if (myid != 0) {
        for (l = 0; l < nw; ++l) {
          MPI_Send(&x[ne+l][i], ib, MPI_DOUBLE, myid-1, i, MPI_COMM_WORLD);
        }
      }
    }
  }
}

// spc
void spc(double A[N][N], double b[M][N], double x[M][N], int n, int m) 
{
     int i, j, k, ne;
     double dtemp;

     int ib = (n + numprocs - 1) / numprocs;
     int i_start = myid * ib;
     int i_end = (myid + 1) * ib > n ? n : (myid + 1) * ib;

     /* LU decomposition ---------------------- */  
     LU(A, n);
     /* --------------------------------------- */

     double C[n][BWIDTH];
     int MAX_NE = (m / BWIDTH) * BWIDTH;
     for (ne=0; ne<MAX_NE; ne+=BWIDTH) {
  
       /* Forward substitution ------------------ */  
       forward(A, b, C, n, m, ne, BWIDTH);
       /* --------------------------------------- */

       /* Backward substitution ------------------ */  
       backward(A, C, x, n, m, ne, BWIDTH);
       /* --------------------------------------- */

     }
     for (; ne < m; ++ne) {
       forward(A, b, C, n, m, ne, 1);
       backward(A, C, x, n, m, ne, 1);
     }
     /* End of m loop ----------------------------------------- */ 

}


