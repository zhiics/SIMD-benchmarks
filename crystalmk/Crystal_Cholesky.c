/*BHEADER****************************************************************
 * (c) 2007   The Regents of the University of California               *
 *                                                                      *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright       *
 * notice and disclaimer.                                               *
 *                                                                      *
 *EHEADER****************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "Crystal.h"

#include <nmmintrin.h>
#include <xmmintrin.h>
#include <immintrin.h>

//#define SSE
//#define AVX

//-------------- 
//  test Cholesky solver on matrix
//-------------- 
void Crystal_Cholesky(int nSlip,  
                      double a[MS_XTAL_NSLIP_MAX][MS_XTAL_NSLIP_MAX],
                      double r[MS_XTAL_NSLIP_MAX],
                      double g[MS_XTAL_NSLIP_MAX])
{
   int i, j, k;
   double fdot;

      /* transfer rhs to solution vector to preserve rhs */
   for ( i = 0; i < nSlip; i++) g[i] = r[i];
      
      /* matrix reduction */
   for ( i = 1; i < nSlip; i++)
      a[i][0] = a[i][0] / a[0][0];


   for ( i = 1; i < nSlip; i++){
      fdot = 0.0;
#if defined(SSE)
   	  __m128d fdot_vec; 
	  fdot_vec = _mm_setzero_pd();
	  for ( k = 0; k < i - i % 2; k += 2){
		  fdot_vec = _mm_add_pd(fdot_vec, _mm_mul_pd(_mm_loadu_pd(&a[i][k]), _mm_set_pd(a[k][i], a[k+1][i])));
	  }
	  fdot_vec = _mm_hadd_pd(fdot_vec, fdot_vec);
	  fdot = _mm_cvtsd_f64(fdot_vec);

      for ( ; k < i; k++)
         fdot += a[i][k] * a[k][i];
#elif defined(AVX)
	  __m256d fdot_vec; 
	  fdot_vec = _mm256_setzero_pd();
	  for ( k = 0; k < i - i % 4; k += 4){
		  fdot_vec = _mm256_add_pd(fdot_vec, _mm256_mul_pd(_mm256_loadu_pd(&a[i][k]), _mm256_set_pd(a[k][i], a[k+1][i], a[k+2][i], a[k+3][i])));
	  }
	  fdot_vec = _mm256_hadd_pd(fdot_vec, fdot_vec);
	  fdot_vec = _mm256_hadd_pd(fdot_vec, fdot_vec);
	  fdot = _mm_cvtsd_f64(_mm256_extractf128_pd(fdot_vec, 0));

      for ( ; k < i; k++)
         fdot += a[i][k] * a[k][i];
#else
      for ( k = 0; k < i; k++)
         fdot += a[i][k] * a[k][i];
#endif
      a[i][i] = a[i][i] - fdot;
      for ( j = i+1; j < nSlip; j++){
         fdot = 0.0;
#if defined(SSE)
		 fdot_vec = _mm_setzero_pd();
		 for ( k = 0; k < i - i % 2; k +=2 ){
		 	fdot_vec = _mm_add_pd(fdot_vec, _mm_mul_pd(_mm_loadu_pd(&a[i][k]), _mm_set_pd(a[k][j], a[k+1][j])));
		 }
	  	 fdot_vec = _mm_hadd_pd(fdot_vec, fdot_vec);
	  	 fdot = _mm_cvtsd_f64(fdot_vec);
         
		 for ( ; k < i; k++)
            fdot += a[i][k] * a[k][j];
#elif defined(AVX)
		 fdot_vec = _mm256_setzero_pd();
		 for ( k = 0; k < i - i % 4; k +=4 ){
		 	fdot_vec = _mm256_add_pd(fdot_vec, _mm256_mul_pd(_mm256_loadu_pd(&a[i][k]), _mm256_set_pd(a[k][j], a[k+1][j], a[k+2][j], a[k+3][j])));
		 }
	  	 fdot_vec = _mm256_hadd_pd(fdot_vec, fdot_vec);
	  	 fdot = _mm_cvtsd_f64(_mm256_extractf128_pd(fdot_vec, 0));
         
		 for ( ; k < i; k++)
            fdot += a[i][k] * a[k][j];

#else
         for ( k = 0; k < i; k++)
            fdot += a[i][k] * a[k][j];
#endif
         a[i][j] = a[i][j] - fdot;
         fdot = 0.0;
#if defined(SSE)
		 fdot_vec = _mm_setzero_pd();
		 for ( k = 0 ; k < i - i % 2; k += 2){
		 	fdot_vec = _mm_add_pd(fdot_vec, _mm_mul_pd(_mm_loadu_pd(&a[j][k]), _mm_set_pd(a[k][i], a[k+1][i])));
		 }

	  	 fdot_vec = _mm_hadd_pd(fdot_vec, fdot_vec);
	  	 fdot = _mm_cvtsd_f64(fdot_vec);
         
		 for ( ; k < i; k++)
            fdot += a[j][k] * a[k][i];
#elif defined(AVX)
		 fdot_vec = _mm256_setzero_pd();
		 for ( k = 0 ; k < i - i % 4; k += 4){
		 	fdot_vec = _mm256_add_pd(fdot_vec, _mm256_mul_pd(_mm256_loadu_pd(&a[j][k]), _mm256_set_pd(a[k][i], a[k+2][i], a[k+1][i], a[k+3][i])));
		 }

	  	 fdot_vec = _mm256_hadd_pd(fdot_vec, fdot_vec);
	  	 fdot = _mm_cvtsd_f64(_mm256_extractf128_pd(fdot_vec, 0));
         
		 for ( ; k < i; k++)
            fdot += a[j][k] * a[k][i];
#else
         for ( k = 0; k < i; k++)
            fdot += a[j][k] * a[k][i];
#endif
         a[j][i] = ( a[j][i] - fdot) / a[i][i];
      }
   }

   
      /* forward reduction of RHS */
   for ( i = 1; i < nSlip; i++ ){
#if defined(SSE)
   	  __m128d fdot_vec; 
	  fdot_vec = _mm_set1_pd(g[i]);
	  for ( k = 0; k < i - i % 2; k += 2) {
		  fdot_vec = _mm_sub_pd(fdot_vec, _mm_mul_pd(_mm_loadu_pd(&a[i][k]), _mm_loadu_pd(&g[k])));
	  }
	  fdot_vec = _mm_hadd_pd(fdot_vec, fdot_vec);
	  g[i] = _mm_cvtsd_f64(fdot_vec);
      
	  for ( ; k < i; k++)
         g[i] = g[i] - a[i][k] * g[k];
#elif defined(AVX)
   	  __m256d fdot_vec; 
	  fdot_vec = _mm256_set1_pd(g[i]);
	  for ( k = 0; k < i - i % 4; k += 4) {
		  fdot_vec = _mm256_sub_pd(fdot_vec, _mm256_mul_pd(_mm256_loadu_pd(&a[i][k]), _mm256_loadu_pd(&g[k])));
	  }
	  fdot_vec = _mm256_hadd_pd(fdot_vec, fdot_vec);
	  fdot_vec = _mm256_hadd_pd(fdot_vec, fdot_vec);
	  g[i] = _mm_cvtsd_f64(_mm256_extractf128_pd(fdot_vec, 0));
      
	  for ( ; k < i; k++)
         g[i] = g[i] - a[i][k] * g[k];
#else
      for ( k = 0; k < i; k++)
         g[i] = g[i] - a[i][k] * g[k];
#endif
   } 
   
      /* back substitution */
   g[nSlip-1] = g[nSlip-1] / a[nSlip-1][nSlip-1];
   for ( i = nSlip - 2; i >= 0; i=i-1){
#if defined(SSE)
   	  __m128d fdot_vec; 
	  fdot_vec = _mm_set1_pd(g[i]);
	  for ( k = i+1; k < nSlip - (nSlip - k) % 2; k += 2) {
		 fdot_vec = _mm_sub_pd(fdot_vec, _mm_mul_pd(_mm_loadu_pd(&a[i][k]), _mm_loadu_pd(&g[k])));
	  }
      
	  fdot_vec = _mm_hadd_pd(fdot_vec, fdot_vec);
	  g[i] = _mm_cvtsd_f64(fdot_vec);
	  
	  for ( ; k < nSlip; k++)
         g[i] = g[i] - a[i][k]*g[k];
#elif defined(AVX)
   	  __m256d fdot_vec; 
	  fdot_vec = _mm256_set1_pd(g[i]);
	  for ( k = i+1; k < nSlip - (nSlip - k) % 4; k += 4) {
		 fdot_vec = _mm256_sub_pd(fdot_vec, _mm256_mul_pd(_mm256_loadu_pd(&a[i][k]), _mm256_loadu_pd(&g[k])));
	  }
      
	  fdot_vec = _mm256_hadd_pd(fdot_vec, fdot_vec);
	  fdot_vec = _mm256_hadd_pd(fdot_vec, fdot_vec);
	  g[i] = _mm_cvtsd_f64(_mm256_extractf128_pd(fdot_vec, 0));
	  
	  for ( ; k < nSlip; k++)
         g[i] = g[i] - a[i][k]*g[k];

#else
      for ( k = i+1; k < nSlip; k++)
         g[i] = g[i] - a[i][k]*g[k];
#endif
      g[i] = g[i] / a[i][i];
   }
}



