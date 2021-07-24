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

     //ierr = MPI_Barrier(MPI_COMM_WORLD);
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


void spc(double A[N][N], double b[M][N], double x[M][N], int n, int m) 
{
     int i, j, k, ne;
     double dtemp;

     int ib = (n + numprocs - 1) / numprocs;
     int i_start = myid * ib;
     int i_end = (myid + 1) * ib > n ? n : (myid + 1) * ib;

     /* LU decomposition ---------------------- */  
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
    //  debug
    //  double At[n][n];
    //  for (int i = 0; i < n; ++i) {
    //    for (int j = 0; j < n; ++j) {
    //      At[i][j] = 0;
    //    }
    //    int is = ib * myid;
    //    int ie = ib * (myid + 1);
    //    for (int j = is; j < ie; ++j) {
    //      At[i][j] = A[i][j];
    //    }
    //  }
    //  MPI_Allreduce(At, A, n*n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
     /* --------------------------------------- */

     i_start = myid * ib;
     for (ne=0; ne<m; ne++) {
       for (i = 0; i < n; ++i) c[i] = 0.0;
       for (i = 0; i < n; ++i) x[ne][i] = 0.0;
  
       /* Forward substitution ------------------ */  
       for (i = i_start; i < n; i += ib) {
         if (myid != 0) {
           MPI_Recv(&c[i], ib, MPI_DOUBLE, myid-1, i, MPI_COMM_WORLD, NULL);
         }
         if (myid == i / ib) {
           for (k = i; k < i + ib; ++k) {
             c[k] = b[ne][k] + c[k];
             for (j = i_start; j < k; ++j) {
               c[k] -= A[k][j] * c[j];
             }
           }
         } else {
           for (k = i; k < i + ib; ++k) {
             for (j = i_start; j < i_end; ++j) {
               c[k] -= A[k][j] * c[j];
             }
           }
           if (myid != numprocs - 1) {
             MPI_Send(&c[i], ib, MPI_DOUBLE, myid+1, i, MPI_COMM_WORLD);
           }
         }
       }
       /* --------------------------------------- */

       /* Backward substitution ------------------ */  
       for (i = i_start; i >= 0; i -= ib) {
         if (myid != numprocs - 1) {
           MPI_Recv(&x[ne][i], ib, MPI_DOUBLE, myid+1, i, MPI_COMM_WORLD, NULL);
         }
         if (i == i_start) {
           for (k = i + ib - 1; k >= i; --k) {
             x[ne][k] = c[k] + x[ne][k];
             for (j = i_end - 1; j > k; --j) {
               x[ne][k] -= A[k][j] * x[ne][j];
             }
             x[ne][k] = x[ne][k] / A[k][k];
           }
         } else {
           for (k = i + ib - 1; k >= i; --k) {
             for (j = i_end - 1; j >= i_start; --j) {
               x[ne][k] -= A[k][j] * x[ne][j];
             }
           }
           if (myid != 0) {
             MPI_Send(&x[ne][i], ib, MPI_DOUBLE, myid-1, i, MPI_COMM_WORLD);
           }
         }
       }
       /* --------------------------------------- */

     }
     /* End of m loop ----------------------------------------- */ 

}


