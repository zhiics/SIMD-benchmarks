// Copyright (c) 2007 Intel Corp.

// Black-Scholes
// Analytical method for calculating European Options
//
// 
// Reference Source: Options, Futures, and Other Derivatives, 3rd Edition, Prentice 
// Hall, John C. Hull,

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <malloc.h>

#ifndef WIN32
#include <pmmintrin.h>
#include <immintrin.h>
#else
#include <xmmintrin.h>
#endif

#ifdef ENABLE_PARSEC_HOOKS
#include <hooks.h>
#endif

// Multi-threaded pthreads header
#ifdef ENABLE_THREADS
// Add the following line so that icc 9.0 is compatible with pthread lib.
#define __thread __threadp  
MAIN_ENV
#undef __thread
#endif

// Multi-threaded OpenMP header
#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

#ifdef ENABLE_TBB
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/tick_count.h"

using namespace std;
using namespace tbb;
#endif //ENABLE_TBB

// Multi-threaded header for Windows
#ifdef WIN32
#pragma warning(disable : 4305)
#pragma warning(disable : 4244)
#include <windows.h>
#endif

#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
#define _MM_ALIGN16 __attribute__((aligned (16)))
#define _MM_ALIGN32 __attribute__((aligned (32)))
#define MUSTINLINE __attribute__((always_inline))
#else
#define MUSTINLINE __forceinline
#define _MM_ALIGN32 __declspec(align(32))
#endif

// NCO = Number of Concurrent Options = SIMD Width
// NCO is currently set in the Makefile.
#ifndef NCO
#error NCO must be defined.
#endif

#if (NCO==4)
#define fptype double
#define SIMD_WIDTH 4
#define _MMR      __m256d
#define _MM_LOAD  _mm256_load_pd
#define _MM_STORE _mm256_store_pd
#define _MM_MUL   _mm256_mul_pd
#define _MM_ADD   _mm256_add_pd
#define _MM_SUB   _mm256_sub_pd
#define _MM_DIV   _mm256_div_pd
#define _MM_SQRT  _mm256_sqrt_pd
#define _MM_SET(A)  _mm256_set1_pd(A)
#define _MM_SETR  _mm256_set_pd
#define _MM_CMPNEQ(A, B) _mm256_cmp_pd(A, B, _CMP_NEQ_UQ)
#define _MM_MOVEMASK _mm256_movemask_pd
#define _MM_CMPLE(A, B) _mm256_cmp_pd(A, B, _CMP_LT_OS)
#define _MM_EXP _mm256_exp_pd
#define _MM_ANDNOT _mm256_andnot_pd
#define _MM_AND _mm256_and_pd
#define _MM_BLENDV _mm256_blendv_pd
#define _MM_LOG _mm256_log_pd
#define _MM_XOR _mm256_xor_pd
#define _MM_FMADD _mm256_fmadd_pd
#define _MM_FMSUB _mm256_fmsub_pd
#endif

#if (NCO==8)
#define fptype float
#define SIMD_WIDTH 8
#define _MMR      __m128
#define _MM_LOAD  _mm_load_ps
#define _MM_STORE _mm_store_ps
#define _MM_MUL   _mm_mul_ps
#define _MM_ADD   _mm_add_ps
#define _MM_SUB   _mm_sub_ps
#define _MM_DIV   _mm_div_ps
#define _MM_SQRT  _mm_sqrt_ps
#define _MM_SET(A)  _mm_set_ps(A,A,A,A)
#define _MM_SETR  _mm_set_ps
#define _MM_CMPNEQ _mm_cmpneq_ps
#define _MM_MOVEMASK _mm_movemask_ps
#endif

#define NUM_RUNS 100

typedef struct __declspec(align(32)) OptionData_ {
        fptype s;          // spot price
        fptype strike;     // strike price
        fptype r;          // risk-free interest rate
        fptype divq;       // dividend rate
        fptype v;          // volatility
        fptype t;          // time to maturity or option expiration in years 
                           //     (1yr = 1.0, 6mos = 0.5, 3mos = 0.25, ..., etc)  
        char OptionType;   // Option type.  "P"=PUT, "C"=CALL
        fptype divs;       // dividend vals (not used in this test)
        fptype DGrefval;   // DerivaGem Reference Value
} OptionData;

_MM_ALIGN32 OptionData* data;
_MM_ALIGN32 fptype* prices;
int numOptions;

//int    * otype;
fptype * otype;
fptype * sptprice;
fptype * strike;
fptype * rate;
fptype * volatility;
fptype * otime;
int numError = 0;
int nThreads;

static int errors = 0;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Cumulative Normal Distribution Function
// See Hull, Section 11.8, P.243-244
#define inv_sqrt_2xPI 0.39894228040143270286

MUSTINLINE void CNDF ( fptype * OutputX, fptype * InputX ) 
{
    _MM_ALIGN32 double sg[SIMD_WIDTH];
    // int sign[SIMD_WIDTH];
    _MMR xSign;
    int i;
    _MMR xInput;
    _MMR xNPrimeofX;
    _MM_ALIGN32 fptype expValues[SIMD_WIDTH];
    _MMR xK2;
    _MMR xK2_2, xK2_3, xK2_4, xK2_5;
    _MMR xLocal, xLocal_1, xLocal_2, xLocal_3, xLocal_n;

//	duplication
#ifdef DUP
    _MMR xInput_dup;
    _MMR xNPrimeofX_dup;
    _MMR xK2_dup;
    _MMR xK2_2_dup, xK2_3_dup, xK2_4_dup, xK2_5_dup;
    _MMR xLocal_dup, xLocal_1_dup, xLocal_2_dup, xLocal_3_dup, xLocal_n_dup;

//	triplication
/*    volatile _MMR xInput_tri;
    volatile _MMR xNPrimeofX_tri;
    volatile _MMR xK2_tri;
    volatile _MMR xK2_2_tri, xK2_3_tri, xK2_4_tri, xK2_5_tri;
    volatile _MMR xLocal_tri, xLocal_1_tri, xLocal_2_tri, xLocal_3_tri;
*/
    _MMR err1, err2, err3, xLocal_bak;
#endif

    /* replaced to use AVX intrinsics */
    /*
    for (i=0; i<SIMD_WIDTH; i++) {
        // Check for negative value of InputX

      if (InputX[i] < 0.0) {
	InputX[i] = -InputX[i];
	sign[i] = 1;
      } else 
	sign[i] = 0;
    }
    */
    xInput = _MM_LOAD(InputX);
    xSign  = _MM_CMPLE(xInput, _MM_SET(0.0));
    xInput = _MM_ANDNOT(_MM_SET(-0.0), xInput);

#ifdef DUP
    xInput_dup = _MM_LOAD(InputX);
    xInput_dup = _MM_ANDNOT(_MM_SET(-0.0), xInput_dup);
    //    xInput_tri = _MM_LOAD(InputX);
#endif
    
// local vars
 
// Compute NPrimeX term common to both four & six decimal accuracy calcs

    /* replaced to use AVX intrinsics */
    //    for (i=0; i<SIMD_WIDTH; i++) {
    //      expValues[i] = exp(-0.5f * InputX[i] * InputX[i]);
    //      printf("exp[%d]: %f\n", i, expValues[i]);
    //    }
    //    xNPrimeofX = _MM_LOAD(expValues);
    
    xNPrimeofX = _MM_EXP(_MM_MUL(_MM_SET(-0.5f), _MM_MUL(xInput, xInput)));
    xNPrimeofX = _MM_MUL(xNPrimeofX, _MM_SET(inv_sqrt_2xPI));
#ifdef DUP
    //    xNPrimeofX_dup = _MM_LOAD(expValues);
    xNPrimeofX_dup = _MM_EXP(_MM_MUL(_MM_SET(-0.5f), _MM_MUL(xInput_dup, xInput_dup)));
    xNPrimeofX_dup = _MM_MUL(xNPrimeofX_dup, _MM_SET(inv_sqrt_2xPI));
//    xNPrimeofX_tri = _MM_LOAD(expValues);
//    xNPrimeofX_tri = _MM_MUL(xNPrimeofX_tri, _MM_SET(inv_sqrt_2xPI));
#endif

    //    xK2 = _MM_MUL(_MM_SET(0.2316419), xInput);
    //    xK2 = _MM_ADD(_MM_SET(1.0), xK2);
    xK2 = _MM_FMADD(_MM_SET(0.2316419), xInput, _MM_SET(1.0)); 
    xK2 = _MM_DIV(_MM_SET(1.0), xK2);
    // xK2 = _mm_rcp_pd(xK2);  // No rcp function for double-precision
#ifdef DUP
    //    xK2_dup = _MM_MUL(_MM_SET(0.2316419), xInput_dup);
    //    xK2_dup = _MM_ADD(_MM_SET(1.0), xK2_dup);
    xK2_dup = _MM_FMADD(_MM_SET(0.2316419), xInput_dup, _MM_SET(1.0)); 
    xK2_dup = _MM_DIV(_MM_SET(1.0), xK2_dup);
//    xK2_tri = _MM_MUL(_MM_SET(0.2316419), xInput_tri);
//    xK2_tri = _MM_ADD(xK2_tri, _MM_SET(1.0));
//    xK2_tri = _MM_DIV(_MM_SET(1.0), xK2_tri);
#endif
    
    xK2_2 = _MM_MUL(xK2, xK2);
    xK2_3 = _MM_MUL(xK2_2, xK2);
    xK2_4 = _MM_MUL(xK2_3, xK2);
    xK2_5 = _MM_MUL(xK2_4, xK2);
#ifdef DUP
    xK2_2_dup = _MM_MUL(xK2_dup, xK2_dup);
    xK2_3_dup = _MM_MUL(xK2_2_dup, xK2_dup);
    xK2_4_dup = _MM_MUL(xK2_3_dup, xK2_dup);
    xK2_5_dup = _MM_MUL(xK2_4_dup, xK2_dup);
//    xK2_2_tri = _MM_MUL(xK2_tri, xK2_tri);
//    xK2_3_tri = _MM_MUL(xK2_2_tri, xK2_tri);
//    xK2_4_tri = _MM_MUL(xK2_3_tri, xK2_tri);
//    xK2_5_tri = _MM_MUL(xK2_4_tri, xK2_tri);
#endif
    
    xLocal_1 = _MM_MUL(xK2, _MM_SET(0.319381530));
    xLocal_2 = _MM_MUL(xK2_2, _MM_SET(-0.356563782));
    //    xLocal_3 = _MM_MUL(xK2_3, _MM_SET(1.781477937));
    //    xLocal_2 = _MM_ADD(xLocal_2, xLocal_3);
    xLocal_2 = _MM_FMADD(xK2_3, _MM_SET(1.781477937), xLocal_2);
    //    xLocal_3 = _MM_MUL(xK2_4, _MM_SET(-1.821255978));
    //    xLocal_2 = _MM_ADD(xLocal_2, xLocal_3);
    xLocal_2 = _MM_FMADD(xK2_4, _MM_SET(-1.821255978), xLocal_2);
    //    xLocal_3 = _MM_MUL(xK2_5, _MM_SET(1.330274429));
    //    xLocal_2 = _MM_ADD(xLocal_2, xLocal_3);
    xLocal_2 = _MM_FMADD(xK2_5, _MM_SET(1.330274429), xLocal_2);
#ifdef DUP
    xLocal_1_dup = _MM_MUL(xK2_dup, _MM_SET(0.319381530));
    xLocal_2_dup = _MM_MUL(xK2_2_dup, _MM_SET(-0.356563782));
    //    xLocal_3_dup = _MM_MUL(xK2_3_dup, _MM_SET(1.781477937));
    //    xLocal_2_dup = _MM_ADD(xLocal_2_dup, xLocal_3_dup);
    xLocal_2_dup = _MM_FMADD(xK2_3_dup, _MM_SET(1.781477937), xLocal_2_dup);
    //    xLocal_3_dup = _MM_MUL(xK2_4_dup, _MM_SET(-1.821255978));
    //    xLocal_2_dup = _MM_ADD(xLocal_2_dup, xLocal_3_dup);
    xLocal_2_dup = _MM_FMADD(xK2_4_dup, _MM_SET(-1.821255978), xLocal_2_dup);
    //    xLocal_3_dup = _MM_MUL(xK2_5_dup, _MM_SET(1.330274429));
    //    xLocal_2_dup = _MM_ADD(xLocal_2_dup, xLocal_3_dup);
    xLocal_2_dup = _MM_FMADD(xK2_5_dup, _MM_SET(1.330274429), xLocal_2_dup);
//    xLocal_1_tri = _MM_MUL(xK2_tri, _MM_SET(0.319381530));
//    xLocal_2_tri = _MM_MUL(xK2_2_tri, _MM_SET(-0.356563782));
//    xLocal_3_tri = _MM_MUL(xK2_3_tri, _MM_SET(1.781477937));
//    xLocal_2_tri = _MM_ADD(xLocal_2_tri, xLocal_3_tri);
//    xLocal_3_tri = _MM_MUL(xK2_4_tri, _MM_SET(-1.821255978));
//    xLocal_2_tri = _MM_ADD(xLocal_2_tri, xLocal_3_tri);
//    xLocal_3_tri = _MM_MUL(xK2_5_tri, _MM_SET(1.330274429));
//    xLocal_2_tri = _MM_ADD(xLocal_2_tri, xLocal_3_tri);
#endif    

    xLocal_1 = _MM_ADD(xLocal_2, xLocal_1);
    xLocal_n = _MM_MUL(xLocal_1, xNPrimeofX);
    xLocal   = _MM_SUB(_MM_SET(1.0), xLocal_n);
#ifdef DUP
    xLocal_1_dup = _MM_ADD(xLocal_2_dup, xLocal_1_dup);
    xLocal_n_dup = _MM_MUL(xLocal_1_dup, xNPrimeofX_dup);
    xLocal_dup   = _MM_SUB(_MM_SET(1.0), xLocal_n_dup);
//    xLocal_1_tri = _MM_ADD(xLocal_2_tri, xLocal_1_tri);
//    xLocal_tri   = _MM_MUL(xLocal_1_tri, xNPrimeofX_tri);
//    xLocal_tri   = _MM_SUB(_MM_SET(1.0), xLocal_tri);
#endif

#ifdef DUP
    err1 = _MM_CMPNEQ(xLocal, xLocal_dup);
    if ( _MM_MOVEMASK(err1) ){
      errors++;
/*		err2 = _MM_CMPNEQ(xLocal, xLocal_tri);
		if ( _MM_MOVEMASK(err2) ){
			errors++;
			err3 = _MM_CMPNEQ(xLocal_dup, xLocal_tri);
			if ( _MM_MOVEMASK(err3) )
				errors++;
			else
				xLocal_bak = xLocal_dup;
		}
		else
			xLocal_bak = xLocal_tri; */
    }
    //	else
    //	xLocal_bak = xLocal;
#endif

    xLocal = _MM_BLENDV(xLocal, xLocal_n, xSign);

//    _MM_STORE(OutputX, xLocal_bak);
    _MM_STORE(OutputX, xLocal);
    // _mm_storel_pd(&OutputX[0], xLocal);
    // _mm_storeh_pd(&OutputX[1], xLocal);

    /* replaced to use AVX intrinsics */
    /*    for (i=0; i<SIMD_WIDTH; i++) {
        if (sign[i]) {
            OutputX[i] = (1.0 - OutputX[i]);
       }
    } */
} 

// For debugging
void print_xmm(_MMR in, char* s) {
    int i;
    _MM_ALIGN32 fptype val[SIMD_WIDTH];

    _MM_STORE(val, in);
    printf("%s: ", s);
    for (i=0; i<SIMD_WIDTH; i++) {
        printf("%f ", val[i]);
    }
    printf("\n");
}

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
void BlkSchlsEqEuroNoDiv (fptype * OptionPrice, int numOptions, fptype * sptprice,
                          fptype * strike, fptype * rate, fptype * volatility,
                          fptype * time, fptype * otype, float timet)
{
    int i;
// local private working variables for the calculation
    _MMR xStockPrice;
    _MMR xStrikePrice;
    _MMR xRiskFreeRate;
    _MMR xVolatility;
    _MMR xTime;
    _MMR xSqrtTime;

    _MM_ALIGN32 fptype logValues[NCO];
    _MMR xLogTerm;
    _MMR xD1, xD2;
    _MMR xPowerTerm;
    _MMR xDen;
    _MM_ALIGN32 fptype d1[SIMD_WIDTH];
    _MM_ALIGN32 fptype d2[SIMD_WIDTH];
    _MM_ALIGN32 fptype FutureValueX[SIMD_WIDTH];
    _MM_ALIGN32 fptype NofXd1[SIMD_WIDTH];
    _MM_ALIGN32 fptype NofXd2[SIMD_WIDTH];
    _MM_ALIGN32 fptype NegNofXd1[SIMD_WIDTH];
    _MM_ALIGN32 fptype NegNofXd2[SIMD_WIDTH];    

    xStockPrice = _MM_LOAD(sptprice);
    xStrikePrice = _MM_LOAD(strike);
    xRiskFreeRate = _MM_LOAD(rate);
    xVolatility = _MM_LOAD(volatility);
    xTime = _MM_LOAD(time);

    xSqrtTime = _MM_SQRT(xTime);

    /* AVX intrinsics are slower than non-AVX */
    for(i=0; i<SIMD_WIDTH;i ++) {
      logValues[i] = log(sptprice[i] / strike[i]);
    }
    xLogTerm = _MM_LOAD(logValues);
    // xLogTerm = _MM_LOG(_MM_DIV(xStockPrice, xStrikePrice));

    xPowerTerm = _MM_MUL(xVolatility, xVolatility);
    xPowerTerm = _MM_MUL(xPowerTerm, _MM_SET(0.5));
    xD1 = _MM_ADD(xRiskFreeRate, xPowerTerm);

    //    xD1 = _MM_MUL(xD1, xTime);

    //    xD1 = _MM_ADD(xD1, xLogTerm);
    xD1 = _MM_FMADD(xD1, xTime, xLogTerm);

    xDen = _MM_MUL(xVolatility, xSqrtTime);
    xD1 = _MM_DIV(xD1, xDen);
    xD2 = _MM_SUB(xD1, xDen);

    _MM_STORE(d1, xD1);
    _MM_STORE(d2, xD2);

    CNDF( NofXd1, d1 );
    CNDF( NofXd2, d2 );

    /* replaced to use AVX intrinsics */
    /*
    for (i=0; i<SIMD_WIDTH; i++) {
        FutureValueX[i] = strike[i] * (exp(-(rate[i])*(time[i])));
        // printf("FV=%lf\n", FutureValueX[i]);

        // NofXd1[i] = NofX(d1[i]);
        // NofXd2[i] = NofX(d2[i]);
        // printf("NofXd1=%lf\n", NofXd1[i]);
        // printf("NofXd2=%lf\n", NofXd2[i]);

        if (otype[i] == 0) {
            OptionPrice[i] = (sptprice[i] * NofXd1[i]) - (FutureValueX[i] * NofXd2[i]);
        }
        else { 
            NegNofXd1[i] = (1.0 - (NofXd1[i]));
            NegNofXd2[i] = (1.0 - (NofXd2[i]));
            OptionPrice[i] = (FutureValueX[i] * NegNofXd2[i]) - (sptprice[i] * NegNofXd1[i]);
        } 
        // printf("OptionPrice[0] = %lf\n", OptionPrice[i]);
    } */

    _MMR xType = _MM_CMPNEQ(_MM_LOAD(otype), _MM_SET(0.0));
    _MMR xNofXd1 = _MM_LOAD(NofXd1);
    _MMR xNofXd2 = _MM_LOAD(NofXd2);
    _MMR xFutureValue = _MM_MUL(xStrikePrice, 
				_MM_EXP(_MM_MUL(_MM_XOR(_MM_SET(-0.0), xRiskFreeRate), xTime)));
    //    _MMR xOptionPrice = _MM_SUB(_MM_MUL(xStockPrice, xNofXd1),
    //				_MM_MUL(xFutureValue, xNofXd2));
    _MMR xTmp         = _MM_MUL(xFutureValue, xNofXd2);
    _MMR xOptionPrice = _MM_FMSUB(xStockPrice, xNofXd1, xTmp);
    _MMR xOptionPrice_else = _MM_ADD(xOptionPrice,
				     _MM_SUB(xFutureValue, xStockPrice));

    xOptionPrice = _MM_BLENDV(xOptionPrice, xOptionPrice_else, xType);
    _MM_STORE(OptionPrice, xOptionPrice);

}

#ifdef ENABLE_TBB
struct mainWork {
  mainWork(){}
  mainWork(mainWork &w, tbb::split){}

  void operator()(const tbb::blocked_range<int> &range) const {
    fptype price[NCO];
    fptype priceDelta;
    int begin = range.begin();
    int end = range.end();

    for (int i=begin; i!=end; i+=NCO) {
      /* Calling main function to calculate option value based on 
       * Black & Scholes's equation.
       */

      BlkSchlsEqEuroNoDiv( price, NCO, &(sptprice[i]), &(strike[i]),
                           &(rate[i]), &(volatility[i]), &(otime[i]), 
                           &(otype[i]), 0);
      for (int k=0; k<NCO; k++) {
        prices[i+k] = price[k];

#ifdef ERR_CHK 
        priceDelta = data[i+k].DGrefval - price[k];
        if( fabs(priceDelta) >= 1e-5 ){
          fprintf(stderr,"Error on %d. Computed=%.5f, Ref=%.5f, Delta=%.5f\n",
                 i+k, price, data[i+k].DGrefval, priceDelta);
          numError ++;
        }
#endif
      }
    }
  }
};

#endif // ENABLE_TBB

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_TBB
int bs_thread(void *tid_ptr) {
    int j;
    tbb::affinity_partitioner a;

    mainWork doall;
    for (j=0; j<NUM_RUNS; j++) {
      tbb::parallel_for(tbb::blocked_range<int>(0, numOptions), doall, a);
    }

    return 0;
}
#else // !ENABLE_TBB

#ifdef WIN32
DWORD WINAPI bs_thread(LPVOID tid_ptr){
#else
int bs_thread(void *tid_ptr) {
#endif
    int i, j, k;
    fptype price[NCO];
    fptype priceDelta;
    int tid = *(int *)tid_ptr;
    int start = tid * (numOptions / nThreads);
    int end = start + (numOptions / nThreads);

    for (j=0; j<NUM_RUNS; j++) {
#ifdef ENABLE_OPENMP
#pragma omp parallel for private(i, price, priceDelta)
        for (i=0; i<numOptions; i += NCO) {
#else  //ENABLE_OPENMP
        for (i=start; i<end; i += NCO) {
#endif //ENABLE_OPENMP
            // Calling main function to calculate option value based on Black & Scholes's
            // equation.
            BlkSchlsEqEuroNoDiv(price, NCO, &(sptprice[i]), &(strike[i]),
                                &(rate[i]), &(volatility[i]), &(otime[i]), &(otype[i]), 0);
            for (k=0; k<NCO; k++) {
              prices[i+k] = price[k];
            } 
#ifdef ERR_CHK
            for (k=0; k<NCO; k++) {
                priceDelta = data[i+k].DGrefval - price[k];
                if (fabs(priceDelta) >= 1e-4) {
                    printf("Error on %d. Computed=%.5f, Ref=%.5f, Delta=%.5f\n",
                           i + k, price[k], data[i+k].DGrefval, priceDelta);
                    numError ++;
                }
            }
#endif
        }
    }

    return 0;
}
#endif //ENABLE_TBB

int main (int argc, char **argv)
{
    FILE *file;
    int i;
    int loopnum;
    fptype * buffer;
    int * buffer2;
    int rv;
	long long start, end;

#ifdef PARSEC_VERSION
#define __PARSEC_STRING(x) #x
#define __PARSEC_XSTRING(x) __PARSEC_STRING(x)
        printf("PARSEC Benchmark Suite Version "__PARSEC_XSTRING(PARSEC_VERSION)"\n");
	fflush(NULL);
#else
        printf("PARSEC Benchmark Suite\n");
	fflush(NULL);
#endif //PARSEC_VERSION
#ifdef ENABLE_PARSEC_HOOKS
   __parsec_bench_begin(__parsec_blackscholes);
#endif

   if (argc != 4)
        {
                printf("Usage:\n\t%s <nthreads> <inputFile> <outputFile>\n", argv[0]);
                exit(1);
        }
    nThreads = atoi(argv[1]);
    char *inputFile = argv[2];
    char *outputFile = argv[3];

    //Read input data from file
    file = fopen(inputFile, "r");
    if(file == NULL) {
      printf("ERROR: Unable to open file `%s'.\n", inputFile);
      exit(1);
    }
    rv = fscanf(file, "%i", &numOptions);
    if(rv != 1) {
      printf("ERROR: Unable to read from file `%s'.\n", inputFile);
      fclose(file);
      exit(1);
    }
    if(NCO > numOptions) {
      printf("ERROR: Not enough work for SIMD operation.\n");
      fclose(file);
      exit(1);
    }
    if(nThreads > numOptions/NCO) {
      printf("WARNING: Not enough work, reducing number of threads to match number of SIMD options packets.\n");
      nThreads = numOptions/NCO;
    }

#if !defined(ENABLE_THREADS) && !defined(ENABLE_OPENMP) && !defined(ENABLE_TBB)
    if(nThreads != 1) {
        printf("Error: <nthreads> must be 1 (serial version)\n");
        exit(1);
    }
#endif

    data = (OptionData*)malloc(numOptions*sizeof(OptionData));
    prices = (fptype*)malloc(numOptions*sizeof(fptype));
    for ( loopnum = 0; loopnum < numOptions; ++ loopnum )
    {
        rv = fscanf(file, "%lf %lf %lf %lf %lf %lf %c %lf %lf", &data[loopnum].s, &data[loopnum].strike, &data[loopnum].r, &data[loopnum].divq, &data[loopnum].v, &data[loopnum].t, &data[loopnum].OptionType, &data[loopnum].divs, &data[loopnum].DGrefval);
        if(rv != 9) {
          printf("ERROR: Unable to read from file `%s'.\n", inputFile);
          fclose(file);
          exit(1);
        }
    }
    rv = fclose(file);
    if(rv != 0) {
      printf("ERROR: Unable to close file `%s'.\n", inputFile);
      exit(1);
    }

#ifdef ENABLE_THREADS
    MAIN_INITENV(,8000000,nThreads);
#endif
    printf("Num of Options: %d\n", numOptions);
    printf("Num of Runs: %d\n", NUM_RUNS);

#define PAD 256
#define LINESIZE 64

    buffer = (fptype *) malloc(5 * numOptions * sizeof(fptype) + PAD);
    sptprice = (fptype *) (((unsigned long long)buffer + PAD) & ~(LINESIZE - 1));
    strike = sptprice + numOptions;
    rate = strike + numOptions;
    volatility = rate + numOptions;
    otime = volatility + numOptions;

    buffer2 = (int *) malloc(numOptions * sizeof(fptype) + PAD);
    otype = (fptype *) (((unsigned long long)buffer2 + PAD) & ~(LINESIZE - 1));

    for (i=0; i<numOptions; i++) {
      //otype[i]      = (data[i].OptionType == 'P') ? 1 : 0;
        otype[i]      = (data[i].OptionType == 'P') ? 1.0: 0.0;
        sptprice[i]   = data[i].s;
        strike[i]     = data[i].strike;
        rate[i]       = data[i].r;
        volatility[i] = data[i].v;
        otime[i]      = data[i].t;
    }

    printf("Size of data: %d\n", numOptions * (sizeof(OptionData) + sizeof(int)));

#ifdef ENABLE_PARSEC_HOOKS
    __parsec_roi_begin();
#endif

	//added by zhi to get time
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	start = ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;

#ifdef ENABLE_THREADS
#ifdef WIN32
    HANDLE *threads;
    int *nums;
    threads = (HANDLE *) malloc (nThreads * sizeof(HANDLE));
    nums = (int *) malloc (nThreads * sizeof(int));

    for(i=0; i<nThreads; i++) {
        nums[i] = i;
        threads[i] = CreateThread(0, 0, bs_thread, &nums[i], 0, 0);
    }
    WaitForMultipleObjects(nThreads, threads, TRUE, INFINITE);
    free(threads);
    free(nums);
#else
    int *tids;
    tids = (int *) malloc (nThreads * sizeof(int));

    for(i=0; i<nThreads; i++) {
        tids[i]=i;
        CREATE_WITH_ARG(bs_thread, &tids[i]);
    }
    WAIT_FOR_END(nThreads);
    free(tids);
#endif //WIN32
#else //ENABLE_THREADS
#ifdef ENABLE_OPENMP
    {
        int tid=0;
        omp_set_num_threads(nThreads);
        bs_thread(&tid);
    }
#else //ENABLE_OPENMP
#ifdef ENABLE_TBB
    tbb::task_scheduler_init init(nThreads);

    int tid=0;
    bs_thread(&tid);
#else //ENABLE_TBB
    //serial version
    int tid=0;
    bs_thread(&tid);
#endif //ENABLE_TBB
#endif //ENABLE_OPENMP
#endif //ENABLE_THREADS

#ifdef ENABLE_PARSEC_HOOKS
    __parsec_roi_end();
#endif

	//added by zhi
	clock_gettime(CLOCK_MONOTONIC, &ts);
	end = ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
	printf ("Total simulation time is: %lld microseconds.\n", end - start);
	printf ("The number of errors is: %d.\n", errors);

    //Write prices to output file
    file = fopen(outputFile, "w");
    if(file == NULL) {
      printf("ERROR: Unable to open file `%s'.\n", outputFile);
      exit(1);
    }
    rv = fprintf(file, "%i\n", numOptions);
    if(rv < 0) {
      printf("ERROR: Unable to write to file `%s'.\n", outputFile);
      fclose(file);
      exit(1);
    }
    for(i=0; i<numOptions; i++) {
      rv = fprintf(file, "%.18f\n", prices[i]);
      if(rv < 0) {
        printf("ERROR: Unable to write to file `%s'.\n", outputFile);
        fclose(file);
        exit(1);
      }
    }
    rv = fclose(file);
    if(rv != 0) {
      printf("ERROR: Unable to close file `%s'.\n", outputFile);
      exit(1);
    }

#ifdef ERR_CHK
    printf("Num Errors: %d\n", numError);
#endif
    //    free(data);
    //    free(prices);

#ifdef ENABLE_PARSEC_HOOKS
    __parsec_bench_end();
#endif

    return 0;
}

