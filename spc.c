#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <mpi.h>

#include "spc.h"

#define  NPROCS   288

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


#include <omp.h>
#include <arm_sve.h>
#include <fj_tool/fipp.h>
#include <fj_tool/fapp.h>

#define USE_PROFILER 1

const int BWIDTH = 8;
const int R = 24;
const int IB = (N + NPROCS - 1) / NPROCS;
const int REDUCE_BUFSIZE = 24;
double Asend[REDUCE_BUFSIZE][IB * R], Arecv[REDUCE_BUFSIZE][IB * R];
double C[N][BWIDTH];
double y[N][BWIDTH];

// LU decomposition
void LU(double A[N][IB], int n, double buf[N]) {
  int i, j, k;
  int ib = (n + numprocs - 1) / numprocs;
  int i_start = myid * ib;
  int i_end = (myid + 1) * ib > n ? n : (myid + 1) * ib;
  double dtemp;

  const int is = myid * ib;
  for (k=0; k<n; k++) {
    int root_id = k / ib;
    if (myid == root_id) {
     dtemp = 1.0 / A[k][k-is];
     for (i=k+1; i<n; i++) {
        A[i][k-is] = A[i][k-is]*dtemp;   
     }
     for (i=k+1; i<n; i++) {
        buf[i] = A[i][k-is];
     }
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
      dtemp = buf[j]; // A[j][k-is];
      for (i=i_start; i<i_end; i++) {
        A[j][i-is] = A[j][i-is] - A[k][i-is]*dtemp; 
      }
    }
  }
}

// forward
void forward(double A[N][N], double b[M][N], double c[N][BWIDTH], int n, int m, int ne, int nw, int myid, int numprocs, MPI_Comm COMM) {
  int i, j, k, l;

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
          double c_val = b[ne][k] + c[k][l];
          for (j = i_start; j < k; ++j) {
            c_val -= A[k][j] * c[j][l];
          }
          c[k][l] = c_val;
        }
      }
    } else {
      for (k = i; k < i + ib; ++k) {
        for (l = 0; l < nw; ++l) {
          double c_val = c[k][l];
          for (j = i_start; j < i_end; ++j) {
            c_val -= A[k][j] * c[j][l];
          }
          c[k][l] = c_val;
        }
      }
      if (myid != numprocs - 1) {
        MPI_Send(&c[i][0], ib*BWIDTH, MPI_DOUBLE, myid+1, i, COMM);
      }
    }
  }
}

// backward
void backward(double A[N][N], double c[N][BWIDTH], double x[M][N], int n, int m, int ne, int nw, double y[N][BWIDTH], int myid, int numprocs, MPI_Comm COMM) {
  int i, j, k, l;

  int ib = (n + numprocs - 1) / numprocs;
  int i_start = myid * ib;
  int i_end = (myid + 1) * ib > n ? n : (myid + 1) * ib;

  for (i = 0; i < n; ++i) {
    for (j = 0; j < nw; ++j) {
      y[i][j] = 0.0;
    }
  }

  for (i = i_start; i >= 0; i -= ib) {
    if (myid != numprocs - 1) {
      MPI_Recv(&y[i][0], ib * BWIDTH, MPI_DOUBLE, myid+1, i+n, COMM, NULL);
    }
    if (i == i_start) {
      for (k = i + ib - 1; k >= i; --k) {
        double ainv = 1.0 / A[k][k];
        for (l = 0; l < nw; ++l) {
          double y_val = c[k][l] + y[k][l];
          for (j = i_end - 1; j > k; --j) {
            y_val -= A[k][j] * y[j][l];
          }
          y[k][l] = y_val * ainv;
        }
      }
    } else {
      for (k = i; k < i + ib; ++k) {
        for (l = 0; l < nw; ++l) {
          double y_val = y[k][l];
          for (j = i_start; j < i_end; ++j) {
            y_val -= A[k][j] * y[j][l];
          }
          y[k][l] = y_val;
        }
      }
      if (myid != 0) {
        MPI_Send(&y[i][0], ib * BWIDTH, MPI_DOUBLE, myid-1, i+n, COMM);
      }
    }
  }

  for (l = 0; l < nw; ++l) {
    for (k = 0; k < n; ++k) {
      x[ne+l][k] = y[k][l];
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

     MPI_Comm MPI_COMM_SPC;
     MPI_Comm MPI_COMM_REV;
     int color = myid % R;
     int key = myid / R;
     MPI_Comm_split(MPI_COMM_WORLD, color, key, &MPI_COMM_SPC);
     MPI_Comm_split(MPI_COMM_WORLD, key, color, &MPI_COMM_REV);

     int i, j, k, ne;

     int ib = (n + numprocs - 1) / numprocs;
     int i_start = myid * ib;
     int i_end = (myid + 1) * ib > n ? n : (myid + 1) * ib;

     double A_local[N][IB];
     for (i = 0; i < n; ++i) {
       for (j = 0; j < ib; ++j) {
         A_local[i][j] = A[i][j+i_start];
       }
     }

     /* LU decomposition ---------------------- */  
#if USE_PROFILER
     fapp_start("LU", 1, 0);
     LU(A_local, n, c);
     fapp_stop("LU", 1, 0);
#else
     LU(A_local, n, c);
#endif
     /* --------------------------------------- */

     for (i = 0; i < n; ++i) {
       for (j = 0; j < ib; ++j) {
         A[i][j+i_start] = A_local[i][j];
       }
     }

     for (k = 0; k < n; k += REDUCE_BUFSIZE) {
       int c0 = ib * R;
       int c1 = ib * color;
       int c2 = ib * key * R;
       
       for (i = 0; i < REDUCE_BUFSIZE; ++i) {
         for (j = 0; j < c0; ++j) {
           Asend[i][j] = 0.0;
         }
       }
       for (i = 0; i < REDUCE_BUFSIZE; ++i) {
         for (j = 0; j < ib; ++j) {
           Asend[i][c1 + j] = A[k+i][i_start + j];
         }
       }
       MPI_Allreduce(Asend, Arecv, REDUCE_BUFSIZE*ib*R, MPI_DOUBLE, MPI_SUM, MPI_COMM_REV);
       for (i = 0; i < REDUCE_BUFSIZE; ++i) {
         for (j = 0; j < c0; ++j) {
           A[k+i][c2 + j] = Arecv[i][j];
         }
       }
     }

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
       backward(A, C, x, n, m, ne, BWIDTH, y, key, numprocs_spc, MPI_COMM_SPC);
       fapp_stop("backward", 1, 0);
#else
       backward(A, C, x, n, m, ne, BWIDTH, y, key, numprocs_spc, MPI_COMM_SPC);
#endif
       /* --------------------------------------- */

     }
     for (ne = m / R * R; ne < m; ++ne) {
       forward(A, b, C, n, m, ne, 1, myid, numprocs, MPI_COMM_WORLD);
       backward(A, C, x, n, m, ne, 1, y, myid, numprocs, MPI_COMM_WORLD);
     }
     /* End of m loop ----------------------------------------- */ 

     if (m / R > 0) {
       for (k = 0; k < R; ++k) {
         int h = m / R;
         int c0 = ib * R;
         int c1 = ib * color;
         int c2 = ib * key * R;
         int c3 = h * color;
         
         for (int l = 0; l < h; l += REDUCE_BUFSIZE) {
           if (color == k) {
             for (i = 0; i < REDUCE_BUFSIZE; ++i) {
               for (j = 0; j < c0; ++j) {
                 Asend[i][j] = x[c3 + l + i][c2 + j];
               }
             }
           } else {
             for (i = 0; i < REDUCE_BUFSIZE; ++i) {
               for (j = 0; j < c0; ++j) {
                 Asend[i][j] = 0.0;
               }
             }
           }
           MPI_Allreduce(Asend, Arecv, REDUCE_BUFSIZE*ib*R, MPI_DOUBLE, MPI_SUM, MPI_COMM_REV);
           for (i = 0; i < REDUCE_BUFSIZE; ++i) {
             for (j = 0; j < ib; ++j) {
               x[h * k + l + i][i_start + j] = Arecv[i][c1 + j];
             }
           }
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


