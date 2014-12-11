/* The Computer Language Benchmarks Game
 * http://benchmarksgame.alioth.debian.org/
 *
 * contributed by Ledrug
 * algorithm is a straight copy from Steve Decker et al's Fortran code
 * with GCC SSE2 intrinsics
 *
 * modified to use more SSE intrinsics on Aug 18, 2014
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <pmmintrin.h>
#include <time.h>
//#define SERIAL

int errors = 0;
double bias[2] = {0.0, 1.0};

/* already replaced with SSE intrinsics in inner loop */
inline double A(int i, int j) {
  return ((i+j) * (i+j+1) / 2.0 + i + 1);
}

double dot(double * v, double * u, int n) {
  int i;
  double sum = 0;
  for (i = 0; i < n; i++)
    sum += v[i] * u[i];
  return sum;
}

#ifdef GCC
__attribute__((optimize("no-tree-vectorize")))
#endif 
void mult_Av_serial(double *v, double *out, const int n) {
	int i;
//#ifdef GCC
//#pragma omp parallel for
//#endif
#ifdef ICC
#pragma novector
#endif
	for ( i = 0; i < n; i++) {
		double sum = 0.0;
		int j;
#ifdef ICC
#pragma novector
#endif
		for (j = 0; j < n; j++) {
			double b = v[j];
			double a = A(i, j);
			sum += b / a;
		}
		out[i] = sum;
	}
}

void mult_Av_intrin(double * v, double * out, const int n) {
  int i;

//#ifdef GCC
//#pragma omp parallel for
//#endif
  __m128d err1;
  __m128d r1 = _mm_set_pd(1.0, 1.0);
  __m128d r2 = _mm_set_pd(0.5, 0.5);
  __m128d r3 = _mm_set_pd(2.0, 2.0);
  __m128d xmm_i = _mm_setzero_pd();
  __m128d xmm_j = _mm_load_pd(bias);
  __m128d xmm_i_1;

  for (i = 0; i < n; i++) {
    __m128d sum = _mm_setzero_pd();
    xmm_i_1 = _mm_add_pd(xmm_i, r1);
    __m128d xmm_i_j = _mm_add_pd(xmm_i, xmm_j);

    int j;
    for (j = 0; j < n; j += 2) {
      __m128d b = _mm_load_pd(v+j);
      /* A(i, j) by using SSE intrinsics */
      /* calc (i+j)(i+j+1) at first */
      //      __m128d a = _mm_set_pd(A(i,j), A(i,j+1));
      __m128d xmm_i_j_i_j_1 = _mm_mul_pd(xmm_i_j, xmm_i_j);
      xmm_i_j_i_j_1 = _mm_add_pd(xmm_i_j_i_j_1, xmm_i_j);
      __m128d a = _mm_mul_pd(xmm_i_j_i_j_1, r2);
      a = _mm_add_pd(a, xmm_i_1);
      
      /* compute reciprocal of a */
      __m128d ra = _mm_cvtps_pd(_mm_rcp_ps(_mm_cvtpd_ps(a)));
      ra = _mm_sub_pd(_mm_add_pd(ra, ra), _mm_mul_pd(_mm_mul_pd(a, ra), ra));
      ra = _mm_sub_pd(_mm_add_pd(ra, ra), _mm_mul_pd(_mm_mul_pd(a, ra), ra));

      //      sum = _mm_add_pd(sum, _mm_div_pd(b, a));
      sum = _mm_add_pd(sum, _mm_mul_pd(b, ra));
      
      xmm_i_j = _mm_add_pd(xmm_i_j, r3);
   }
	sum = _mm_hadd_pd(sum, sum);
	out[i] = _mm_cvtsd_f64(sum);
//	_mm_store_sd(out+i, sum);
//	out[i] = sum[0];
    xmm_i = xmm_i_1;
  }
}

#ifdef GCC
__attribute__((optimize("no-tree-vectorize")))
#endif
void mult_Atv_serial(double * v, double * out, const int n) {
	int i;
//#ifdef GCC
//#pragma omp parallel for
//#endif
//#pragma novector	//doesn't work for gcc
#ifdef ICC
#pragma novector
#endif
	for (i = 0; i < n; i++) {
		double sum = 0.0;
		int j;
#ifdef ICC
#pragma novector
#endif
		for (j = 0; j < n; j++) {
			double b = v[j];
			double a = A(i, j);
			sum += b / a;
		}
		out[i] = sum;
	}
}

void mult_Atv_intrin(double * v, double * out, const int n) {
  int i;
//#ifdef GCC
//#pragma omp parallel for
//#endif

  __m128d r1 = _mm_set_pd(1.0, 1.0);
  __m128d r2 = _mm_set_pd(0.5, 0.5);
  __m128d r3 = _mm_set_pd(2.0, 2.0);
  __m128d xmm_i = _mm_setzero_pd();
  __m128d xmm_j = _mm_load_pd(bias);

  for (i = 0; i < n; i++) {
    __m128d sum = _mm_setzero_pd();
    __m128d xmm_i_j = _mm_add_pd(xmm_i, xmm_j);
    __m128d xmm_j_1 = _mm_add_pd(xmm_j, r1);

    int j;
    for (j = 0; j < n; j += 2) {
      __m128d b = _mm_load_pd(v+j);
      /* A(i, j) by using SSE intrinsics */
      /* calc (i+j)(i+j+1) at first */
      //      __m128d a = _mm_set_pd(A(j,i), A(j+1,i));
      __m128d xmm_i_j_i_j_1 = _mm_mul_pd(xmm_i_j, xmm_i_j);
      xmm_i_j_i_j_1 = _mm_add_pd(xmm_i_j_i_j_1, xmm_i_j);
      __m128d a = _mm_mul_pd(xmm_i_j_i_j_1, r2);
      a = _mm_add_pd(a, xmm_j_1);
      
      /* compute reciprocal of a */
      __m128d ra = _mm_cvtps_pd(_mm_rcp_ps(_mm_cvtpd_ps(a)));
      ra = _mm_sub_pd(_mm_add_pd(ra, ra), _mm_mul_pd(_mm_mul_pd(a, ra), ra));
      ra = _mm_sub_pd(_mm_add_pd(ra, ra), _mm_mul_pd(_mm_mul_pd(a, ra), ra));

      //      sum = _mm_add_pd(sum, _mm_div_pd(b, a));
      sum = _mm_add_pd(sum, _mm_mul_pd(b, ra));

      xmm_i_j = _mm_add_pd(xmm_i_j, r3);
      xmm_j_1 = _mm_add_pd(xmm_j_1, r3);
    }
	sum = _mm_hadd_pd(sum, sum);
	out[i] = _mm_cvtsd_f64(sum);
//	_mm_store_sd(out+i, sum);
//	out[i] = sum[0];
    xmm_i = _mm_add_pd(xmm_i, r1);
  }
}

double *tmp;
void mult_AtAv_intrin(double *v, double *out, const int n) {
  mult_Av_intrin(v, tmp, n);
  mult_Atv_intrin(tmp, out, n);
}

void mult_AtAv_serial(double *v, double *out, const int n) {
  mult_Av_serial(v, tmp, n);
  mult_Atv_serial(tmp, out, n);
}

int main(int argc, char**argv) {
  int n = atoi(argv[1]);
  if (n <= 0) n = 2000;
  if (n & 1) n++;   // make it multiple of two

  double *u, *v;
  u = memalign(16, n * sizeof(double));
  v = memalign(16, n * sizeof(double));
  tmp = memalign(16, n * sizeof(double));

  int i;
  long long start, end;
  struct timespec ts;

  clock_gettime(CLOCK_MONOTONIC, &ts);
  start = ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;

#ifdef SERIAL
  for (i = 0; i < n; i++) u[i] = 1;
  for (i = 0; i < 10; i++) {
    mult_AtAv_serial(u, v, n);
    mult_AtAv_serial(v, u, n);
  }

  clock_gettime(CLOCK_MONOTONIC, &ts);
  end = ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;

  printf ("Serial simluation time is %lld microseconds.\n", end - start);
  printf("%.9f\n", sqrt(dot(u,v, n) / dot(v,v,n)));
#else
  clock_gettime(CLOCK_MONOTONIC, &ts);
  start = ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;

  for (i = 0; i < n; i++) u[i] = 1;
  for (i = 0; i < 10; i++) {
    mult_AtAv_intrin(u, v, n);
    mult_AtAv_intrin(v, u, n);
  }

  clock_gettime(CLOCK_MONOTONIC, &ts);
  end = ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;

  printf ("Intrinsic simluation time is %lld microseconds.\n", end - start);
  printf("%.9f\n", sqrt(dot(u,v, n) / dot(v,v,n)));
#endif  
  printf ("Total errors: %d\n", errors);

  return 0;
}
