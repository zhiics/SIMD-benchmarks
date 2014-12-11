#include <stdio.h> 
#include <math.h> 

#include <nmmintrin.h>
#include <immintrin.h>
/* fft on a set of n points given by A_re and A_im. Bit-reversal permuted roots-of-unity lookup table
 * is given by W_re and W_im. More specifically,  W is the array of first n/2 nth roots of unity stored
 * in a permuted bitreversal order.
 *
 * FFT - Decimation In Time FFT with input array in correct order and output array in bit-reversed order.
 *
 * REQ: n should be a power of 2 to work. 
 *
 * Note: - See www.cs.berkeley.edu/~randit for her thesis on VIRAM FFTs and other details about VHALF section of the algo
 *         (thesis link - http://www.cs.berkeley.edu/~randit/papers/csd-00-1106.pdf)
 *       - See the foll. CS267 website for details of the Decimation In Time FFT implemented here.
 *         (www.cs.berkeley.edu/~demmel/cs267/lecture24/lecture24.html)
 *       - Also, look "Cormen Leicester Rivest [CLR] - Introduction to Algorithms" book for another variant of Iterative-FFT
 */

extern int errors;

void fft_serial(int n, double *A_re, double *A_im, double *W_re, double *W_im) 
{
  double w_re, w_im, u_re, u_im, t_re, t_im;
  int m, g, b;
  int i, mt, k;

  /* for each stage */  
  for (m=n; m>=2; m=m>>1) 
  {
    /* m = n/2^s; mt = m/2; */
    mt = m >> 1;

    /* for each group of butterfly */ 
    for (g=0,k=0; g<n; g+=m,k++) 
    {
      /* each butterfly group uses only one root of unity. actually, it is the bitrev of this group's number k.
       * BUT 'bitrev' it as a log2n-1 bit number because we are using a lookup array of nth root of unity and
       * using cancellation lemma to scale nth root to n/2, n/4,... th root.
       *
       * It turns out like the foll.
       *   w.re = W[bitrev(k, log2n-1)].re;
       *   w.im = W[bitrev(k, log2n-1)].im;
       * Still, we just use k, because the lookup array itself is bit-reversal permuted. 
       */
      w_re = W_re[k];
      w_im = W_im[k];

      /* for each butterfly */ 
      for (b=g; b<(g+mt); b++) 
      {
        /* printf("bf %d %d %d %f %f %f %f\n", m, g, b, A_re[b], A_im[b], A_re[b+mt], A_im[b+mt]);
         */ 
        //printf("bf %d %d %d (u,t) %g %g %g %g (w) %g %g\n", m, g, b, A_re[b], A_im[b], A_re[b+mt], A_im[b+mt], w_re, w_im);

        /* t = w * A[b+mt] */
        t_re = w_re * A_re[b+mt] - w_im * A_im[b+mt];
        t_im = w_re * A_im[b+mt] + w_im * A_re[b+mt];
		printf("%d, %f, %f\n", b, t_re, t_im);

        /* u = A[b]; in[b] = u + t; in[b+mt] = u - t; */
        u_re = A_re[b];
        u_im = A_im[b];
        A_re[b] = u_re + t_re;
        A_im[b] = u_im + t_im;
        A_re[b+mt] = u_re - t_re;
        A_im[b+mt] = u_im - t_im;

        /*  printf("af %d %d %d %f %f %f %f\n", m, g, b, A_re[b], A_im[b], A_re[b+mt], A_im[b+mt]);
         */         
        //printf("af %d %d %d (u,t) %g %g %g %g (w) %g %g\n", m, g, b, A_re[b], A_im[b], A_re[b+mt], A_im[b+mt], w_re, w_im);
      }
    }
  }
}

void fft(int n, double *A_re, double *A_im, double *W_re, double *W_im) 
{
  double w_re, w_im, u_re, u_im, t_re, t_im;
  int m, g, b;
  int i, mt, k;

  __m128d w_re_vec, w_im_vec, t_re_vec, t_im_vec, u_re_vec, u_im_vec, A_re_vec, A_im_vec, err1, err2, err3;

  for (m=n; m>=2; m=m>>1) 
  {
    mt = m >> 1;
    for (g=0,k=0; g<n; g+=m,k++) 
    {
	  	w_re_vec = _mm_set1_pd(W_re[k]);
	  	w_im_vec = _mm_set1_pd(W_im[k]);

		if (mt%2) {
      		w_re = W_re[k];
      		w_im = W_im[k];
		}

      for (b=g; b<(g+mt) - mt%2; b+=2) 
      {
		  A_re_vec = _mm_load_pd(A_re+mt+b);
		  A_im_vec = _mm_load_pd(A_im+mt+b);
		  t_re_vec = _mm_sub_pd(_mm_mul_pd(w_re_vec, A_re_vec), _mm_mul_pd(w_im_vec, A_im_vec));
		  t_im_vec = _mm_add_pd(_mm_mul_pd(w_re_vec, A_im_vec), _mm_mul_pd(w_im_vec, A_re_vec));
		  double *v1 = (double *) &t_re_vec;
		  double *v2 = (double *) &t_im_vec;
		  
		  u_re_vec = _mm_load_pd(A_re+b);
		  u_im_vec = _mm_load_pd(A_im+b);
		  _mm_store_pd(A_re+b, _mm_add_pd(u_re_vec, t_re_vec));
		  _mm_store_pd(A_im+b, _mm_add_pd(u_im_vec, t_im_vec));
		  _mm_store_pd(A_re+mt+b, _mm_sub_pd(u_re_vec, t_re_vec));
		  _mm_store_pd(A_im+mt+b, _mm_sub_pd(u_im_vec, t_im_vec));

      }

      for (; b<(g+mt); b++) 
      {
        t_re = w_re * A_re[b+mt] - w_im * A_im[b+mt];
        t_im = w_re * A_im[b+mt] + w_im * A_re[b+mt];

        u_re = A_re[b];
        u_im = A_im[b];
        A_re[b] = u_re + t_re;
        A_im[b] = u_im + t_im;
        A_re[b+mt] = u_re - t_re;
        A_im[b+mt] = u_im - t_im;
      }

    }
  }
}

void fft_avx(int n, double *A_re, double *A_im, double *W_re, double *W_im) 
{
  double w_re, w_im, u_re, u_im, t_re, t_im;
  int m, g, b;
  int i, mt, k;

  __m256d w_re_avx, w_im_avx, t_re_avx, t_im_avx, u_re_avx, u_im_avx, A_re_avx, A_im_avx, err1, err2, err3;

  for (m=n; m>=2; m=m>>1) 
  {
    mt = m >> 1;
    for (g=0,k=0; g<n; g+=m,k++) 
    {
//	  	w_re_avx = _mm256_set1_pd(W_re[k]);
	  	w_re_avx = _mm256_broadcast_sd(&W_re[k]);
//	  	w_im_avx = _mm256_set1_pd(W_im[k]);
	  	w_im_avx = _mm256_broadcast_sd(&W_im[k]);

		if (mt%4) {
      		w_re = W_re[k];
      		w_im = W_im[k];
		}

      for (b=g; b<(g+mt) - mt%4; b+=4) 
      {
		  A_re_avx = _mm256_load_pd(A_re+mt+b);
		  A_im_avx = _mm256_load_pd(A_im+mt+b);
		  t_re_avx = _mm256_sub_pd(_mm256_mul_pd(w_re_avx, A_re_avx), _mm256_mul_pd(w_im_avx, A_im_avx));
		  t_im_avx = _mm256_add_pd(_mm256_mul_pd(w_re_avx, A_im_avx), _mm256_mul_pd(w_im_avx, A_re_avx));
		  
		  u_re_avx = _mm256_load_pd(A_re+b);
		  u_im_avx = _mm256_load_pd(A_im+b);
		  _mm256_store_pd(A_re+b, _mm256_add_pd(u_re_avx, t_re_avx));
		  _mm256_store_pd(A_im+b, _mm256_add_pd(u_im_avx, t_im_avx));
		  _mm256_store_pd(A_re+mt+b, _mm256_sub_pd(u_re_avx, t_re_avx));
		  _mm256_store_pd(A_im+mt+b, _mm256_sub_pd(u_im_avx, t_im_avx));

      }

      for (; b<(g+mt); b++) 
      {
        t_re = w_re * A_re[b+mt] - w_im * A_im[b+mt];
        t_im = w_re * A_im[b+mt] + w_im * A_re[b+mt];

        u_re = A_re[b];
        u_im = A_im[b];
        A_re[b] = u_re + t_re;
        A_im[b] = u_im + t_im;
        A_re[b+mt] = u_re - t_re;
        A_im[b+mt] = u_im - t_im;
      }

    }
  }
}

