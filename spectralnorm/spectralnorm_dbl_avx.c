/* The Computer Language Benchmarks Game
 * http://benchmarksgame.alioth.debian.org/
 *
 * contributed by Ledrug
 * algorithm is a straight copy from Steve Decker et al's Fortran code
 * with GCC SSE2 intrinsics
 *
 * modified to use AVX intrinsics on Aug 18, 2014
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <pmmintrin.h>
#include <immintrin.h>
#include <time.h>

int errors = 0;
double bias[4] = {0.0, 1.0, 2.0, 3.0};
double half = 0.5;
double one = 1.0;
double four = 4.0;

/* already replaced with AVX intrinsics in inner loop */
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
  __m256d err1;
  __m256d r1 = _mm256_set_pd(1.0, 1.0, 1.0, 1.0);
  __m256d r2 = _mm256_set_pd(0.5, 0.5, 0.5, 0.5);
  __m256d r3 = _mm256_set_pd(4.0, 4.0, 4.0, 4.0);
  __m256d ymm_i = _mm256_setzero_pd();
  __m256d ymm_j = _mm256_load_pd(bias);
  __m256d ymm_i_1;

  for (i = 0; i < n; i++) {
    ymm_i_1 = _mm256_add_pd(ymm_i, r1);
    __m256d ymm_i_j = _mm256_add_pd(ymm_i, ymm_j);
    __m256d sum = _mm256_setzero_pd();

    int j;
    for (j = 0; j < n; j += 4) {
      __m256d b = _mm256_load_pd(v+j);

      /* A(i, j) by using AVX intrinsics */
      /* calc (i+j)(i+j+1) at first */
      // volatile __m256d a = _mm256_setr_pd(A(i,j), A(i,j+1), A(i,j+2), A(i, j+3));
      __m256d ymm_i_j_i_j_1 = _mm256_mul_pd(ymm_i_j, ymm_i_j);
      ymm_i_j_i_j_1 = _mm256_add_pd(ymm_i_j_i_j_1, ymm_i_j);
      __m256d a = _mm256_mul_pd(ymm_i_j_i_j_1, r2);
      a = _mm256_add_pd(a, ymm_i_1);

      /* compute reciprocal of a */
      __m256d ra = _mm256_cvtps_pd(_mm_rcp_ps(_mm256_cvtpd_ps(a)));
      ra = _mm256_sub_pd(_mm256_add_pd(ra, ra), _mm256_mul_pd(_mm256_mul_pd(a, ra), ra));
      ra = _mm256_sub_pd(_mm256_add_pd(ra, ra), _mm256_mul_pd(_mm256_mul_pd(a, ra), ra));

      //      sum = _mm256_add_pd(sum, _mm256_div_pd(b, a));
      sum = _mm256_add_pd(sum, _mm256_mul_pd(b, ra));

      ymm_i_j = _mm256_add_pd(ymm_i_j, r3);
    }
    sum = _mm256_hadd_pd(sum, sum);

    __m128d sum_high = _mm256_extractf128_pd(sum, 1);
    __m128d hadd = _mm_add_pd(sum_high, _mm256_castpd256_pd128(sum));
    out[i] = _mm_cvtsd_f64(hadd);
    //	_mm_store_pd(out+i, val);
    ymm_i = ymm_i_1;
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
  __m256d err1;
  __m256d r1 = _mm256_set_pd(1.0, 1.0, 1.0, 1.0);
  __m256d r2 = _mm256_set_pd(0.5, 0.5, 0.5, 0.5);
  __m256d r3 = _mm256_set_pd(4.0, 4.0, 4.0, 4.0);
  __m256d ymm_i = _mm256_setzero_pd();
  __m256d ymm_j = _mm256_load_pd(bias);

  for (i = 0; i < n; i++) {
    __m256d sum = _mm256_setzero_pd();

    int j;
    __m256d ymm_i_j = _mm256_add_pd(ymm_i, ymm_j);
    __m256d ymm_j_1 = _mm256_add_pd(ymm_j, r1);
    for (j = 0; j < n; j += 4) {
      __m256d b = _mm256_load_pd(v+j);

      /* A(i, j) by using AVX intrinsics */
      /* calc (i+j)(i+j+1) at first */
      // volatile __m256d a = _mm256_setr_pd(A(j,i), A(j+1,i), A(j+2, i), A(j+3, i));
      __m256d ymm_i_j_i_j_1 = _mm256_mul_pd(ymm_i_j, ymm_i_j);
      ymm_i_j_i_j_1 = _mm256_add_pd(ymm_i_j_i_j_1, ymm_i_j);
      __m256d a = _mm256_mul_pd(ymm_i_j_i_j_1, r2);
      a = _mm256_add_pd(a, ymm_j_1);

      /* compute reciprocal of a */
      __m256d ra = _mm256_cvtps_pd(_mm_rcp_ps(_mm256_cvtpd_ps(a)));
      ra = _mm256_sub_pd(_mm256_add_pd(ra, ra), _mm256_mul_pd(_mm256_mul_pd(a, ra), ra));
      ra = _mm256_sub_pd(_mm256_add_pd(ra, ra), _mm256_mul_pd(_mm256_mul_pd(a, ra), ra));

      //      sum = _mm256_add_pd(sum, _mm256_div_pd(b, a));
      sum = _mm256_add_pd(sum, _mm256_mul_pd(b, ra));

      ymm_i_j = _mm256_add_pd(ymm_i_j, r3);
      ymm_j_1 = _mm256_add_pd(ymm_j_1, r3);
    }
    sum = _mm256_hadd_pd(sum, sum);

    __m128d sum_high = _mm256_extractf128_pd(sum, 1);
    __m128d hadd = _mm_add_pd(sum_high, _mm256_castpd256_pd128(sum));
    out[i] = _mm_cvtsd_f64(hadd);
    //	_mm_store_pd(out+i, hadd);
    //	out[i] = sum[0];
    
    ymm_i = _mm256_add_pd(ymm_i, r1);
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
  u = memalign(32, n * sizeof(double));
  v = memalign(32, n * sizeof(double));
  tmp = memalign(32, n * sizeof(double));

  int i;
  long long start, end;
  struct timespec ts;

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
  printf ("Total errors: %d\n", errors);

  return 0;
}
