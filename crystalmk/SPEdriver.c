/*BHEADER****************************************************************
 * (c) 2007   The Regents of the University of California               *
 *                                                                      *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright       *
 * notice and disclaimer.                                               *
 *                                                                      *
 *EHEADER****************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include "Crystal.h"

int errors = 0;

void Crystal_div(int nSlip,  
                double deltaTime,
                double slipRate[MS_XTAL_NSLIP_MAX],
                double dSlipRate[MS_XTAL_NSLIP_MAX],
                double tau[MS_XTAL_NSLIP_MAX],
                double tauc[MS_XTAL_NSLIP_MAX],
                double rhs[MS_XTAL_NSLIP_MAX],
                double dtcdgd[MS_XTAL_NSLIP_MAX][MS_XTAL_NSLIP_MAX],
	        double dtdg[MS_XTAL_NSLIP_MAX][MS_XTAL_NSLIP_MAX],
	        double matrix[MS_XTAL_NSLIP_MAX][MS_XTAL_NSLIP_MAX]);

double Crystal_pow(int nSlip,  
		  double slipRate[MS_XTAL_NSLIP_MAX]);


void Crystal_Cholesky(int nSlip,  
                     double a[MS_XTAL_NSLIP_MAX][MS_XTAL_NSLIP_MAX],
                     double r[MS_XTAL_NSLIP_MAX],
                     double g[MS_XTAL_NSLIP_MAX]);

//-------------- 
//  SPE driver
//-------------- 
double SPEdriver(double slipRate[MS_XTAL_NSLIP_MAX],
                 double dSlipRate[MS_XTAL_NSLIP_MAX],
                 double tau[MS_XTAL_NSLIP_MAX],
                 double tauc[MS_XTAL_NSLIP_MAX],
                 double rhs[MS_XTAL_NSLIP_MAX],
                 double dtcdgd[MS_XTAL_NSLIP_MAX][MS_XTAL_NSLIP_MAX],
	         double dtdg[MS_XTAL_NSLIP_MAX][MS_XTAL_NSLIP_MAX],
	         double matrix[MS_XTAL_NSLIP_MAX][MS_XTAL_NSLIP_MAX])
{
  struct timeval  t0, t1;
  clock_t t0_cpu = 0,
          t1_cpu = 0;

  double del_wtime = 0.0;
  double tmp;

  const int noIter = 1000000;
  int i = 0;
  int j, k;


  gettimeofday(&t0, ((void *)0));
  t0_cpu = clock();
  for (i=0; i<noIter; ++i) {
      Crystal_div(MS_XTAL_NSLIP_MAX,  
                 0.01,
                 slipRate,
                 dSlipRate,
                 tau,
                 tauc,
                 rhs,
                 dtcdgd,
                 dtdg,
                 matrix);
  }
  gettimeofday(&t1, ((void *)0)); 
  t1_cpu = clock();

  del_wtime = (double)(t1.tv_sec - t0.tv_sec) +
              (double)(t1.tv_usec - t0.tv_usec)/1000000.0;


  printf("\n***** timing for Crystal_div \n");  

  printf("\nWall time = %f seconds. \n", del_wtime);

  printf("\nCPU  time = %f seconds. \n\n", ((double) (t1_cpu - t0_cpu))/CLOCKS_PER_SEC);


  gettimeofday(&t0, ((void *)0));
  t0_cpu = clock();
  for (i=0; i<noIter; ++i) {
       tmp = Crystal_pow(MS_XTAL_NSLIP_MAX,
	     slipRate);
  }
  gettimeofday(&t1, ((void *)0)); 
  t1_cpu = clock();

  del_wtime = (double)(t1.tv_sec - t0.tv_sec) +
              (double)(t1.tv_usec - t0.tv_usec)/1000000.0;


  printf("\n***** timing for Crystal_pow \n");  

  printf("\nWall time = %f seconds. \n", del_wtime);

  printf("\nCPU  time = %f seconds. \n\n", ((double) (t1_cpu - t0_cpu))/CLOCKS_PER_SEC);


  gettimeofday(&t0, ((void *)0));
  t0_cpu = clock();

  for (i=0; i<MS_XTAL_NSLIP_MAX; i++){
    for (j=0; j<MS_XTAL_NSLIP_MAX; j++)
      matrix[i][j] = dtcdgd[i][j];
  }

  for (i=0; i<noIter; ++i) { 
       for (j=0; j<MS_XTAL_NSLIP_MAX; j++){
         for (k=0; k<MS_XTAL_NSLIP_MAX; k++)
           dtcdgd[j][k] = matrix[j][k];
       }
       Crystal_Cholesky(MS_XTAL_NSLIP_MAX,  
                       dtcdgd,
                       tau,
                       rhs);
  }
  gettimeofday(&t1, ((void *)0)); 
  t1_cpu = clock();

  del_wtime = (double)(t1.tv_sec - t0.tv_sec) +
              (double)(t1.tv_usec - t0.tv_usec)/1000000.0;


  printf("\n***** timing for Crystal_Cholesky \n");  

  printf("\nWall time = %f seconds. \n", del_wtime);

  printf("\nCPU  time = %f seconds. \n\n", ((double) (t1_cpu - t0_cpu))/CLOCKS_PER_SEC);
  printf("Number of errors is: %d\n", errors);
  return tmp;

}
