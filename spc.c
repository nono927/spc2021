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
#include <fj_tool/fipp.h>
#include <fj_tool/fapp.h>

#define USE_PROFILER 1

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
    int root_id = k / ib;
    if (myid == root_id) {
     dtemp = 1.0 / A[k][k];
     for (i=k+1; i<n; i++) {
        A[i][k] = A[i][k]*dtemp;   
     }
     for (i=k+1; i<n; i++) {
        buf[i] = A[i][k];
     }
    //  for (int dst_id = myid + 1; dst_id < numprocs; ++dst_id) {
    //    MPI_Send(buf, n, MPI_DOUBLE, dst_id, k, MPI_COMM_WORLD);
    //  }
     int dst1 = root_id + 1;
     int dst2 = root_id + 2;
     if (dst1 < numprocs) MPI_Send(&buf[k+1], n-k-1, MPI_DOUBLE, dst1, k, MPI_COMM_WORLD);
     if (dst2 < numprocs) MPI_Send(&buf[k+1], n-k-1, MPI_DOUBLE, dst2, k, MPI_COMM_WORLD);
     ++i_start;
    } else if (root_id < myid) {
      int d = myid - root_id + 1;
      int src  = root_id + (d >> 1) - 1;
      int dst1 = root_id + (d << 1) - 1;
      int dst2 = root_id + (d << 1);
      MPI_Recv(&buf[k+1], n-k-1, MPI_DOUBLE, src, k, MPI_COMM_WORLD, NULL);
      if (dst1 < numprocs) MPI_Send(&buf[k+1], n-k-1, MPI_DOUBLE, dst1, k, MPI_COMM_WORLD);
      if (dst2 < numprocs) MPI_Send(&buf[k+1], n-k-1, MPI_DOUBLE, dst2, k, MPI_COMM_WORLD);
    } else {
      break;
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
void forward(double A[N][N], double b[M][N], double c[N][BWIDTH], int n, int m, int ne, int nw, int myid, int numprocs, MPI_Comm COMM) {
  int i, j, k, l;
  double dtemp;

  int ib = (n + numprocs - 1) / numprocs;
  int i_start = myid * ib;
  int i_end = (myid + 1) * ib > n ? n : (myid + 1) * ib;

  for (i = 0; i < n; ++i) 
    for (l = 0; l < nw; ++l) c[i][l] = 0.0;

  for (i = i_start; i < n; i += ib) {
    if (myid != 0) {
      MPI_Recv(&c[i][0], ib*BWIDTH, MPI_DOUBLE, myid-1, i, COMM, NULL);
    }
    if (myid == i / ib) {
      for (k = i; k < i + ib; ++k) {
        for (l = 0; l < nw; ++l) {
          // c[k][l] = b[ne][k] + c[k][l];
          double c_val = b[ne][k] + c[k][l];
          for (j = i_start; j < k; ++j) {
            // c[k][l] -= A[k][j] * c[j][l];
            c_val -= A[k][j] * c[j][l];
          }
          c[k][l] = c_val;
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
        MPI_Send(&c[i][0], ib*BWIDTH, MPI_DOUBLE, myid+1, i, COMM);
      }
    }
  }
}

// backward
void backward(double A[N][N], double c[N][BWIDTH], double x[M][N], int n, int m, int ne, int nw, double xbuf[], int myid, int numprocs, MPI_Comm COMM) {
  int i, j, k, l;
  double dtemp;

  int ib = (n + numprocs - 1) / numprocs;
  int i_start = myid * ib;
  int i_end = (myid + 1) * ib > n ? n : (myid + 1) * ib;

  // double xbuf[nw * ib];

  for (i = 0; i < n; ++i) x[ne][i] = 0.0;

  for (i = i_start; i >= 0; i -= ib) {
    if (myid != numprocs - 1) {
      MPI_Recv(xbuf, nw * ib, MPI_DOUBLE, myid+1, i+n, COMM, NULL);
      for (l = 0; l < nw; ++l) {
        for (j = 0; j < ib; ++j) {
          x[ne+l][i+j] = xbuf[ib * l + j];
        }
      }
    }
    if (i == i_start) {
      for (l = 0; l < nw; ++l) {
        for (k = i + ib - 1; k >= i; --k) {
          double x_val = c[k][l] + x[ne+l][k];
          // x[ne+l][k] = c[k][l] + x[ne+l][k];
          for (j = i_end - 1; j > k; --j) {
            // x[ne+l][k] -= A[k][j] * x[ne+l][j];
            x_val -= A[k][j] * x[ne+l][j];
          }
          // x[ne+l][k] = x[ne+l][k] / A[k][k];
          x[ne+l][k] = x_val / A[k][k];
        }
      }
    } else {
      for (l = 0; l < nw; ++l) {
        for (k = i + ib - 1; k >= i; --k) {
          double x_val = x[ne+l][k];
          for (j = i_end - 1; j >= i_start; --j) {
            // x[ne+l][k] -= A[k][j] * x[ne+l][j];
            x_val -= A[k][j] * x[ne+l][j];
          }
          x[ne+l][k] = x_val;
        }
      }
      if (myid != 0) {
        for (l = 0; l < nw; ++l) {
          for (j = 0; j < ib; ++j) {
            xbuf[ib * l + j] = x[ne+l][i+j];
          }
        }
        MPI_Send(xbuf, nw * ib, MPI_DOUBLE, myid-1, i+n, COMM);
      }
    }
  }
}

// spc
void spc(double A[N][N], double b[M][N], double x[M][N], int n, int m) 
{
#if USE_PROFILER
     fipp_start();
     fapp_start("spc", 1, 0);
#endif

     const int R = 24;
     MPI_Comm MPI_COMM_SPC;
     MPI_Comm MPI_COMM_REV;
     int color = myid % R;
     int key = myid / R;
     MPI_Comm_split(MPI_COMM_WORLD, color, key, &MPI_COMM_SPC);
     MPI_Comm_split(MPI_COMM_WORLD, key, color, &MPI_COMM_REV);

     int i, j, k, ne;
     double dtemp;

     int ib = (n + numprocs - 1) / numprocs;
     int i_start = myid * ib;
     int i_end = (myid + 1) * ib > n ? n : (myid + 1) * ib;

     /* LU decomposition ---------------------- */  
#if USE_PROFILER
     fapp_start("LU", 1, 0);
     LU(A, n);
     fapp_stop("LU", 1, 0);
#else
     LU(A, n);
#endif
     /* --------------------------------------- */

     {
       int c0 = ib * R;
       int c1 = ib * color;
       int c2 = ib * key * R;
       double Asend[n][c0];
       double Arecv[n][c0];
       for (i = 0; i < n; ++i) {
         for (j = 0; j < c0; ++j) {
           Asend[i][j] = 0.0;
         }
       }
       for (i = 0; i < n; ++i) {
         for (j = 0; j < ib; ++j) {
           Asend[i][c1 + j] = A[i][i_start + j];
         }
       }
       MPI_Allreduce(Asend, Arecv, n*ib*R, MPI_DOUBLE, MPI_SUM, MPI_COMM_REV);
       for (i = 0; i < n; ++i) {
         for (j = 0; j < ib * R; ++j) {
           A[i][c2 + j] = Arecv[i][j];
         }
       }
     }

     double C[n][BWIDTH];
     double xbuf[BWIDTH*ib*R];
     int MAX_NE = (m / BWIDTH) * BWIDTH;
     int ne_start = (m / R) * color;
     int ne_end = (m / R) * (color + 1);
     int numprocs_spc = numprocs / R;

     for (ne=ne_start; ne<ne_end; ne+=BWIDTH) {
  
       /* Forward substitution ------------------ */
#if USE_PROFILER
       fapp_start("forward", 1, 0);  
       forward(A, b, C, n, m, ne, BWIDTH, key, numprocs_spc, MPI_COMM_SPC);
       fapp_stop("forward", 1, 0);
#else
       forward(A, b, C, n, m, ne, BWIDTH, key, numprocs_spc, MPI_COMM_SPC);
#endif
       /* --------------------------------------- */

       /* Backward substitution ------------------ */  
#if USE_PROFILER
       fapp_start("backward", 1, 0);
       backward(A, C, x, n, m, ne, BWIDTH, xbuf, key, numprocs_spc, MPI_COMM_SPC);
       fapp_stop("backward", 1, 0);
#else
       backward(A, C, x, n, m, ne, BWIDTH, xbuf, key, numprocs_spc, MPI_COMM_SPC);
#endif
       /* --------------------------------------- */

     }
     for (ne = m / R * R; ne < m; ++ne) {
       forward(A, b, C, n, m, ne, 1, myid, numprocs, MPI_COMM_WORLD);
       backward(A, C, x, n, m, ne, 1, xbuf, myid, numprocs, MPI_COMM_WORLD);
     }
     /* End of m loop ----------------------------------------- */ 

     {
       int h = m / R;
       int c0 = ib * R;
       int c1 = ib * color;
       int c2 = ib * key * R;
       int c3 = h * color;
       double xsend[m][ib * R];
       double xrecv[m][ib * R];
       for (i = 0; i < m; ++i) {
         for (j = 0; j < c0; ++j) {
           xsend[i][j] = 0.0;
         }
       }
       for (i = 0; i < h; ++i) {
         for (j = 0; j < c0; ++j) {
           xsend[c3 + i][j] = x[c3 + i][c2 + j];
         }
       }
       MPI_Allreduce(xsend, xrecv, m*ib*R, MPI_DOUBLE, MPI_SUM, MPI_COMM_REV);
       for (i = 0; i < m; ++i) {
         for (j = 0; j < ib; ++j) {
           x[i][i_start + j] = xrecv[i][c1 + j];
         }
       }
     }

     MPI_Comm_free(&MPI_COMM_SPC);
     MPI_Comm_free(&MPI_COMM_REV);
#if USE_PROFILER
     fapp_stop("spc", 1, 0);
     fipp_stop();
#endif
}


