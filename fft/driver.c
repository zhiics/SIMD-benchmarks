#include <stdio.h> 
#include <math.h>   
#include <assert.h> 

#include <time.h>
#include <nmmintrin.h>

/************************************************************************************/

void init_array(int n, double *A_re, double *A_im); 
void compute_W(int n, double *W_re, double *W_im); 
void output_array(int n, double *A_re, double *A_im, char *outfile); 
void permute_bitrev(int n, double *A_re, double *A_im); 
int bitrev(int inp, int numbits); 
int log_2(int n);  

int errors = 0;

/************************************************************************************/


/* gets no. of points from the user, initialize the points and roots of unity lookup table 
 * and lets fft go. finally bit-reverses the results and outputs them into a file. 
 * n should be a power of 2. 
 */ 
int main(int argc, char *argv[])
{
  int n; 
  double *A_re, *A_im, *W_re, *W_im; 
  long long start, end;
  struct timespec ts;

  if (argc <= 2) 
  {
    fprintf(stderr, "Usage: ./fft n outfile\n"); 
    exit(-1); 
  }
  n = atoi(argv[1]); 
 
  A_re = (double*)_mm_malloc(sizeof(double)*n, 32); 
  A_im = (double*)_mm_malloc(sizeof(double)*n, 32); 
  W_re = (double*)_mm_malloc(sizeof(double)*n/2, 32); 
  W_im = (double*)_mm_malloc(sizeof(double)*n/2, 32); 
  assert(A_re != NULL && A_im != NULL && W_re != NULL && W_im != NULL); 

  init_array(n, A_re, A_im); 
  compute_W(n, W_re, W_im); 
  clock_gettime(CLOCK_MONOTONIC, &ts);
  start = ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
  fft(n, A_re, A_im, W_re, W_im);
  clock_gettime(CLOCK_MONOTONIC, &ts);
  end = ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
  permute_bitrev(n, A_re, A_im);        
 // output_array(n, A_re, A_im, argv[2]);  
  
  printf ("The total CPU time of FFT is: %lld microseconds.\n", end - start);
  printf ("The number of errors is %lld \n.", errors);

  _mm_free(A_re); 
  _mm_free(A_im); 
  _mm_free(W_re); 
  _mm_free(W_im); 
}

/* initializes the array with some function of n */  
void init_array(int n, double *A_re, double *A_im) 
{
  int NumPoints, i;
  NumPoints     = 0;

  #ifdef COMMENT_ONLY 
  for(i=0; i < n*2 ; i+=2)
  {
    A_re[NumPoints] = (double)input_buf[i];  
    A_im[NumPoints] = (double)input_buf[i+1];  
    /* printf("%d,%d -> %g,%g\n", input_buf[i], input_buf[i+1], A_re[NumPoints], A_im[NumPoints]); */  
    /* printf("%g %g\n", A_re[NumPoints], A_im[NumPoints]);  */  
    NumPoints++;
  }
  #endif 

  for (i=0; i<n; i++)
  {
      if (i==1) 
      {
        A_re[i]=1.0; 
        A_im[i]=0.0; 
      }  
      else
      {
        A_re[i]=0.0; 
        A_im[i]=0.0; 
      } 
      #ifdef COMMENT_ONLY 
      A_re[i] = sin_lookup[i];  /* sin((double)i*2*M_PI/(double)n); */  
      A_im[i] = sin_lookup[i];  /* sin((double)i*2*M_PI/(double)n); */  
      #endif 
  } 
  //A_re[255] = 1.0;  
   
} 


/* W will contain roots of unity so that W[bitrev(i,log2n-1)] = e^(2*pi*i/n)
 * n should be a power of 2
 * Note: W is bit-reversal permuted because fft(..) goes faster if this is done.
 *       see that function for more details on why we treat 'i' as a (log2n-1) bit number.
 */
void compute_W(int n, double *W_re, double *W_im)
{
  int i, br;
  int log2n = log_2(n);

  for (i=0; i<(n/2); i++)
  {
    br = bitrev(i,log2n-1); 
    W_re[br] = cos(((double)i*2.0*M_PI)/((double)n));  
    W_im[br] = sin(((double)i*2.0*M_PI)/((double)n));  
  }
  #ifdef COMMENT_ONLY 
  for (i=0;i<(n/2);i++)
  { 
    br = i; //bitrev(i,log2n-1); 
    printf("(%g\t%g)\n", W_re[br], W_im[br]);
  }  
  #endif 
}


/* a file with name "outfile" will be opened.  It's first line will have the value of n.  Subsequent lines will
 * contain entries of A. 
 */
void output_array(int n, double *A_re, double *A_im, char *outfile)
{
  int i;
  FILE *fp;
  fp = fopen(outfile, "w");
  if (fp == NULL) {
    printf("could not open %s for output\n", outfile);
    exit(-1);
  }
  /* fprintf(fp, "%d\n", n); */ 
  for (i=0; i<n; i++)
    fprintf(fp, "%.16f %.16f\n", A_re[i], A_im[i]);  
    /* fprintf(fp, "%lf %lf\n", A_re[i]/(double)n, A_im[i]/(double)n); */  
  fclose(fp);
}


/* permutes the array using a bit-reversal permutation */ 
void permute_bitrev(int n, double *A_re, double *A_im) 
{ 
  int i, bri, log2n;
  double t_re, t_im;

  log2n = log_2(n); 
  
  for (i=0; i<n; i++)
  {
      bri = bitrev(i, log2n);

      /* skip already swapped elements */
      if (bri <= i) continue;

      t_re = A_re[i];
      t_im = A_im[i];
      A_re[i]= A_re[bri];
      A_im[i]= A_im[bri];
      A_re[bri]= t_re;
      A_im[bri]= t_im;
  }  
} 


/* treats inp as a numbits number and bitreverses it. 
 * inp < 2^(numbits) for meaningful bit-reversal
 */ 
int bitrev(int inp, int numbits)
{
  int i, rev=0;
  for (i=0; i < numbits; i++)
  {
    rev = (rev << 1) | (inp & 1);
    inp >>= 1;
  }
  return rev;
}


/* returns log n (to the base 2), if n is positive and power of 2 */ 
int log_2(int n) 
{
  int res; 
  for (res=0; n >= 2; res++) 
    n = n >> 1; 
  return res; 
}
