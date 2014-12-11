/**
Copyright 2012 the Generic SIMD Intrinsic Library project authors. All rights reserved.

Copyright IBM Corp. 2013, 2013. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
     * Redistributions in binary form must reproduce the above
     copyright notice, this list of conditions and the following
     disclaimer in the documentation and/or other materials provided
     with the distribution.
     * Neither the name of IBM Corp. nor the names of its contributors may be
     used to endorse or promote products derived from this software
     without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <getopt.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <smmintrin.h>
#include <xmmintrin.h>
#include <algorithm>
#include <time.h>
#include <inttypes.h>
/*
  g++ -I../../include mandelbrot.cc -mvsx -flax-vector-conversions -Wno-int-to-pointer-cast -O3 -o mandelbrot
 */

 /* 
                Scalar version of mandelbrot 
 */
static int mandel(double c_re, double c_im, int count) {
  double z_re = c_re, z_im = c_im;
  int cci=0;
  for (cci = 0; cci < count; ++cci) {
    if (z_re * z_re + z_im * z_im > 4.f)
      break;

    double new_re = z_re*z_re - z_im*z_im;
    double new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
  }
  return cci;
 }

void mandelbrot_serial(double x0, double y0, double x1, double y1,
                       int width, int height, int maxIterations,
                       int output[])
{
  double dx = (x1 - x0) / width;
  double dy = (y1 - y0) / height;

  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; ++i) {
      double x = x0 + i * dx;
      double y = y0 + j * dy;

      int index = (j * width + i);
      output[index] = mandel(x, y, maxIterations);
    }
  }
}

static __m128i mandelSSE(__m128d c_re, __m128d c_im, int count) {
	__m128d z_re = c_re;
	__m128d z_im = c_im;
	__m128d cons2 = _mm_set1_pd(2.0);
	__m128d cons4 = _mm_set1_pd(4.0);
	__m128d mask = _mm_set1_pd(0xffffffffffffffff);

	int cci = 0;
	int check;
	__m128i ret = _mm_set1_epi64x((int64_t)cci);

	for (cci = 0; cci < count; ++cci) {
		__m128i tmp = _mm_set1_epi64x((int64_t)cci);

		ret = _mm_castpd_si128(_mm_blendv_pd(_mm_castsi128_pd(ret), _mm_castsi128_pd(tmp), mask));
		mask = _mm_cmple_pd(_mm_add_pd(_mm_mul_pd(z_re, z_re), _mm_mul_pd(z_im, z_im)), cons4);

		check = _mm_movemask_pd(mask);
		if (!check)
			break;

		__m128d new_re = _mm_sub_pd(_mm_mul_pd(z_re, z_re), _mm_mul_pd(z_im, z_im));
		__m128d new_im = _mm_mul_pd(_mm_mul_pd(cons2, z_re), z_im);
		z_re = _mm_add_pd(c_re, new_re);
		z_im = _mm_add_pd(c_im, new_im);
	}

	if (cci == count) {
		__m128i tmp = _mm_set1_epi64x((int64_t)cci);
		ret = _mm_castpd_si128(_mm_blendv_pd(_mm_castsi128_pd(ret), _mm_castsi128_pd(tmp), mask));
	}

	return ret;
}

/*
              Generic Intrinsics
*/
void mandelbrot_generic(double x0, double y0, double x1, double y1,
			int width, int height, int maxIterations,
			int output[])
{

  double dx = (x1 - x0) / width;
  double dy = (y1 - y0) / height;

  __m128d v_x0 = _mm_set1_pd(x0);
  __m128d v_y0 = _mm_set1_pd(y0);
  
  __m128d v_dx = _mm_set1_pd(dx);
  __m128d v_dy = _mm_set1_pd(dy);

  __m128d v_i, v_j;
  __m128i out; 

  for (int j = 0; j < height; j++) {
    v_j = _mm_set1_pd((double) j); 
    for (int i = 0; i < width; i+=2) {
      //float x = x0 + i * dx;
      //float y = y0 + j * dy;
	  v_i = _mm_set_pd((double) (i+1), (double) i);
      __m128d v_x = _mm_add_pd(v_x0, _mm_mul_pd(v_i, v_dx));
      __m128d v_y = _mm_add_pd(v_y0, _mm_mul_pd(v_j, v_dy));
      
      int index = (j * width + i);
      //output[index] = mandel(x, y, maxIterations);
	  out = mandelSSE(v_x, v_y, maxIterations);
	  int64_t *val = (int64_t *) &out;
	  output[index] = val[0];
	  output[index+1] = val[1];
//	  _mm_storeu_si128((__m128i *) (output + index), out);
    }    
  }
}


/* Write a PPM image file with the image of the Mandelbrot set */
static void
writePPM(int *buf, int width, int height, const char *fn) {
  FILE *fp = fopen(fn, "wb");
  fprintf(fp, "P6\n");
  fprintf(fp, "%d %d\n", width, height);
  fprintf(fp, "255\n");
  for (int i = 0; i < width*height; ++i) {
    // Map the iteration count to colors by just alternating between
    // two greys.
    char c = (buf[i] & 0x1) ? 240 : 20;
    for (int j = 0; j < 3; ++j)
      fputc(c, fp);
  }
  fclose(fp);
  printf("Wrote image file %s\n", fn);
}


static void
writePPM_d(int *buf, int width, int height, const char *fn) {
  for (int i = 0; i < width; ++i) {
    for (int j = 0; j < height; ++j) {
      int index = i*width+j;
      printf("%4d ", buf[index]); 
    }
    printf("\n");
  }
  printf("Wrote image file %s\n", fn);
}


int main() {
//  unsigned int width = 768;
//  unsigned int height = 512;

//  unsigned int width = 1024;
//  unsigned int height = 1024;
  
  unsigned int width = 4096 * 4;
  unsigned int height = 4096 * 4;

  double x0 = -2;
  double x1 = 1;
  double y0 = -1;
  double y1 = 1;

  int maxIterations = 10;
  int *buf = new int[width*height];
  
  long long start, end;
  struct timespec ts;
  //
  // Compute the image using the scalar and generic intrinsics implementations; report the minimum
  // time of three runs.
  //
/*  
  clock_gettime(CLOCK_MONOTONIC, &ts);
  start = ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
  for (int i = 0; i < 3; ++i) {
    mandelbrot_serial(x0, y0, x1, y1, width, height, maxIterations, buf);
  }
  clock_gettime(CLOCK_MONOTONIC, &ts);
  end = ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
  printf("[mandelbrot serial]:\t\t[%lld] microseconds\n", end - start);
 // writePPM(buf, width, height, "mandelbrot-serial-dbl.ppm");
  printf("buf[%d] = %d\n", width, buf[width]);
*/
  clock_gettime(CLOCK_MONOTONIC, &ts);
  start = ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
  for (int i = 0; i < 3; ++i) {
    mandelbrot_generic(x0, y0, x1, y1, width, height, maxIterations, buf);
  }
  clock_gettime(CLOCK_MONOTONIC, &ts);
  end = ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
  printf("[mandelbrot intrin]:\t\t[%lld] microseconds\n", end - start);
  printf("buf[%d] = %d\n", width, buf[width]);
//  writePPM(buf, width, height, "mandelbrot-sse-dbl.ppm");

  return 0;

}
