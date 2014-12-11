/*BHEADER****************************************************************
 * (c) 2006   The Regents of the University of California               *
 *                                                                      *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright       *
 * notice and disclaimer.                                               *
 *                                                                      *
 *EHEADER****************************************************************/

#include "irsmk.h"
#include <xmmintrin.h>
#include <immintrin.h>

extern int errors;

void rmatmult3(Domain_t *domain, RadiationData_t *rblk, double *x, double *b )
{
   char *me = "rmatmult3" ;
   int i, ii, jj, kk, tt ;
   int imin = domain->imin ;
   int imax = domain->imax ;
   int jmin = domain->jmin ;
   int jmax = domain->jmax ;
   int kmin = domain->kmin ;
   int kmax = domain->kmax ;
   int jp   = domain->jp   ;
   int kp   = domain->kp   ;
   double *dbl = rblk->dbl ;
   double *dbc = rblk->dbc ;
   double *dbr = rblk->dbr ;
   double *dcl = rblk->dcl ;
   double *dcc = rblk->dcc ;
   double *dcr = rblk->dcr ;
   double *dfl = rblk->dfl ;
   double *dfc = rblk->dfc ;
   double *dfr = rblk->dfr ;
   double *cbl = rblk->cbl ;
   double *cbc = rblk->cbc ;
   double *cbr = rblk->cbr ;
   double *ccl = rblk->ccl ;
   double *ccc = rblk->ccc ;
   double *ccr = rblk->ccr ;
   double *cfl = rblk->cfl ;
   double *cfc = rblk->cfc ;
   double *cfr = rblk->cfr ;
   double *ubl = rblk->ubl ;
   double *ubc = rblk->ubc ;
   double *ubr = rblk->ubr ;
   double *ucl = rblk->ucl ;
   double *ucc = rblk->ucc ;
   double *ucr = rblk->ucr ;
   double *ufl = rblk->ufl ;
   double *ufc = rblk->ufc ;
   double *ufr = rblk->ufr ;
   double *xdbl = x - kp - jp - 1 ;
   double *xdbc = x - kp - jp     ;
   double *xdbr = x - kp - jp + 1 ;
   double *xdcl = x - kp      - 1 ;
   double *xdcc = x - kp          ;
   double *xdcr = x - kp      + 1 ;
   double *xdfl = x - kp + jp - 1 ;
   double *xdfc = x - kp + jp     ;
   double *xdfr = x - kp + jp + 1 ;
   double *xcbl = x      - jp - 1 ;
   double *xcbc = x      - jp     ;
   double *xcbr = x      - jp + 1 ;
   double *xccl = x           - 1 ;
   double *xccc = x               ;
   double *xccr = x           + 1 ;
   double *xcfl = x      + jp - 1 ;
   double *xcfc = x      + jp     ;
   double *xcfr = x      + jp + 1 ;
   double *xubl = x + kp - jp - 1 ;
   double *xubc = x + kp - jp     ;
   double *xubr = x + kp - jp + 1 ;
   double *xucl = x + kp      - 1 ;
   double *xucc = x + kp          ;
   double *xucr = x + kp      + 1 ;
   double *xufl = x + kp + jp - 1 ;
   double *xufc = x + kp + jp     ;
   double *xufr = x + kp + jp + 1 ;

   __m128d RA, RB, RC;
//   volatile __m128d RA1, RB1, RC1, RC_bak;
//   volatile __m128d RA2, RB2, RC2;
   __m128d err1, err2, err3;

   for ( kk = kmin ; kk < kmax ; kk++ ) {
      for ( jj = jmin ; jj < jmax ; jj++ ) {
//         for ( ii = imin ; ii < imax ; ii++ ) {
	       for ( ii = imin ; ii < imax ; ii+=2 ) {
	    	i = ii + jj * jp + kk * kp ;

//            b[i] = dbl[i] * xdbl[i] + dbc[i] * xdbc[i] + dbr[i] * xdbr[i] + dcl[i] * xdcl[i] + \
				   dcc[i] * xdcc[i] + dcr[i] * xdcr[i] + dfl[i] * xdfl[i] + dfc[i] * xdfc[i] + \
				   dfr[i] * xdfr[i] + cbl[i] * xcbl[i] + cbc[i] * xcbc[i] + cbr[i] * xcbr[i] + \
                   ccl[i] * xccl[i] + ccc[i] * xccc[i] + ccr[i] * xccr[i] + cfl[i] * xcfl[i] + \
				   cfc[i] * xcfc[i] + cfr[i] * xcfr[i] + ubl[i] * xubl[i] + ubc[i] * xubc[i] + \
				   ubr[i] * xubr[i] + ucl[i] * xucl[i] + ucc[i] * xucc[i] + ucr[i] * xucr[i] + \
                   ufl[i] * xufl[i] + ufc[i] * xufc[i] + ufr[i] * xufr[i] ;

			RC = _mm_setzero_pd(); 
			RA = _mm_loadu_pd(&dbl[i]);	RB = _mm_loadu_pd(&xdbl[i]); RC = _mm_add_pd(RC, _mm_mul_pd(RA, RB));
//			_mm_storeu_pd(test, RA);
			RA = _mm_loadu_pd(&dbc[i]);	RB = _mm_loadu_pd(&xdbc[i]); RC = _mm_add_pd(RC, _mm_mul_pd(RA, RB));
			RA = _mm_loadu_pd(&dbr[i]);	RB = _mm_loadu_pd(&xdbr[i]); RC = _mm_add_pd(RC, _mm_mul_pd(RA, RB));
			
			RA = _mm_loadu_pd(&dcl[i]);	RB = _mm_loadu_pd(&xdcl[i]); RC = _mm_add_pd(RC, _mm_mul_pd(RA, RB));
			RA = _mm_loadu_pd(&dcc[i]);	RB = _mm_loadu_pd(&xdcc[i]); RC = _mm_add_pd(RC, _mm_mul_pd(RA, RB));
			RA = _mm_loadu_pd(&dcr[i]);	RB = _mm_loadu_pd(&xdcr[i]); RC = _mm_add_pd(RC, _mm_mul_pd(RA, RB));
			
			RA = _mm_loadu_pd(&dfl[i]);	RB = _mm_loadu_pd(&xdfl[i]); RC = _mm_add_pd(RC, _mm_mul_pd(RA, RB));
			RA = _mm_loadu_pd(&dfc[i]);	RB = _mm_loadu_pd(&xdfc[i]); RC = _mm_add_pd(RC, _mm_mul_pd(RA, RB));
			RA = _mm_loadu_pd(&dfr[i]);	RB = _mm_loadu_pd(&xdfr[i]); RC = _mm_add_pd(RC, _mm_mul_pd(RA, RB));
			
			RA = _mm_loadu_pd(&cbl[i]);	RB = _mm_loadu_pd(&xcbl[i]); RC = _mm_add_pd(RC, _mm_mul_pd(RA, RB));
			RA = _mm_loadu_pd(&cbc[i]);	RB = _mm_loadu_pd(&xcbc[i]); RC = _mm_add_pd(RC, _mm_mul_pd(RA, RB));
			RA = _mm_loadu_pd(&cbr[i]);	RB = _mm_loadu_pd(&xcbr[i]); RC = _mm_add_pd(RC, _mm_mul_pd(RA, RB));
			
			RA = _mm_loadu_pd(&ccl[i]);	RB = _mm_loadu_pd(&xccl[i]); RC = _mm_add_pd(RC, _mm_mul_pd(RA, RB));
			RA = _mm_loadu_pd(&ccc[i]);	RB = _mm_loadu_pd(&xccc[i]); RC = _mm_add_pd(RC, _mm_mul_pd(RA, RB));
			RA = _mm_loadu_pd(&ccr[i]);	RB = _mm_loadu_pd(&xccr[i]); RC = _mm_add_pd(RC, _mm_mul_pd(RA, RB));
			
			RA = _mm_loadu_pd(&cfl[i]);	RB = _mm_loadu_pd(&xcfl[i]); RC = _mm_add_pd(RC, _mm_mul_pd(RA, RB));
			RA = _mm_loadu_pd(&cfc[i]);	RB = _mm_loadu_pd(&xcfc[i]); RC = _mm_add_pd(RC, _mm_mul_pd(RA, RB));
			RA = _mm_loadu_pd(&cfr[i]);	RB = _mm_loadu_pd(&xcfr[i]); RC = _mm_add_pd(RC, _mm_mul_pd(RA, RB));
			
			RA = _mm_loadu_pd(&ubl[i]);	RB = _mm_loadu_pd(&xubl[i]); RC = _mm_add_pd(RC, _mm_mul_pd(RA, RB));
			RA = _mm_loadu_pd(&ubc[i]);	RB = _mm_loadu_pd(&xubc[i]); RC = _mm_add_pd(RC, _mm_mul_pd(RA, RB));
			RA = _mm_loadu_pd(&ubr[i]);	RB = _mm_loadu_pd(&xubr[i]); RC = _mm_add_pd(RC, _mm_mul_pd(RA, RB));
			
			RA = _mm_loadu_pd(&ucl[i]);	RB = _mm_loadu_pd(&xucl[i]); RC = _mm_add_pd(RC, _mm_mul_pd(RA, RB));
			RA = _mm_loadu_pd(&ucc[i]);	RB = _mm_loadu_pd(&xucc[i]); RC = _mm_add_pd(RC, _mm_mul_pd(RA, RB));
			RA = _mm_loadu_pd(&ucr[i]);	RB = _mm_loadu_pd(&xucr[i]); RC = _mm_add_pd(RC, _mm_mul_pd(RA, RB));
			
			RA = _mm_loadu_pd(&ufl[i]);	RB = _mm_loadu_pd(&xufl[i]); RC = _mm_add_pd(RC, _mm_mul_pd(RA, RB));
			RA = _mm_loadu_pd(&ufc[i]);	RB = _mm_loadu_pd(&xufc[i]); RC = _mm_add_pd(RC, _mm_mul_pd(RA, RB));
			RA = _mm_loadu_pd(&ufr[i]);	RB = _mm_loadu_pd(&xufr[i]); RC = _mm_add_pd(RC, _mm_mul_pd(RA, RB));
	
			_mm_storeu_pd(&b[i], RC); 
	 	}
      }
   }

}

void rmatmult3_avx(Domain_t *domain, RadiationData_t *rblk, double *x, double *b )
{
   char *me = "rmatmult3" ;
   int i, ii, jj, kk, tt ;
   int imin = domain->imin ;
   int imax = domain->imax ;
   int jmin = domain->jmin ;
   int jmax = domain->jmax ;
   int kmin = domain->kmin ;
   int kmax = domain->kmax ;
   int jp   = domain->jp   ;
   int kp   = domain->kp   ;
   double *dbl = rblk->dbl ;
   double *dbc = rblk->dbc ;
   double *dbr = rblk->dbr ;
   double *dcl = rblk->dcl ;
   double *dcc = rblk->dcc ;
   double *dcr = rblk->dcr ;
   double *dfl = rblk->dfl ;
   double *dfc = rblk->dfc ;
   double *dfr = rblk->dfr ;
   double *cbl = rblk->cbl ;
   double *cbc = rblk->cbc ;
   double *cbr = rblk->cbr ;
   double *ccl = rblk->ccl ;
   double *ccc = rblk->ccc ;
   double *ccr = rblk->ccr ;
   double *cfl = rblk->cfl ;
   double *cfc = rblk->cfc ;
   double *cfr = rblk->cfr ;
   double *ubl = rblk->ubl ;
   double *ubc = rblk->ubc ;
   double *ubr = rblk->ubr ;
   double *ucl = rblk->ucl ;
   double *ucc = rblk->ucc ;
   double *ucr = rblk->ucr ;
   double *ufl = rblk->ufl ;
   double *ufc = rblk->ufc ;
   double *ufr = rblk->ufr ;
   double *xdbl = x - kp - jp - 1 ;
   double *xdbc = x - kp - jp     ;
   double *xdbr = x - kp - jp + 1 ;
   double *xdcl = x - kp      - 1 ;
   double *xdcc = x - kp          ;
   double *xdcr = x - kp      + 1 ;
   double *xdfl = x - kp + jp - 1 ;
   double *xdfc = x - kp + jp     ;
   double *xdfr = x - kp + jp + 1 ;
   double *xcbl = x      - jp - 1 ;
   double *xcbc = x      - jp     ;
   double *xcbr = x      - jp + 1 ;
   double *xccl = x           - 1 ;
   double *xccc = x               ;
   double *xccr = x           + 1 ;
   double *xcfl = x      + jp - 1 ;
   double *xcfc = x      + jp     ;
   double *xcfr = x      + jp + 1 ;
   double *xubl = x + kp - jp - 1 ;
   double *xubc = x + kp - jp     ;
   double *xubr = x + kp - jp + 1 ;
   double *xucl = x + kp      - 1 ;
   double *xucc = x + kp          ;
   double *xucr = x + kp      + 1 ;
   double *xufl = x + kp + jp - 1 ;
   double *xufc = x + kp + jp     ;
   double *xufr = x + kp + jp + 1 ;

   __m256d RA, RB, RC, RC1, RC2, RC3;
   RC = _mm256_setzero_pd();

   for ( kk = kmin ; kk < kmax ; kk++ ) {
      for ( jj = jmin ; jj < jmax ; jj++ ) {
	       for ( ii = imin ; ii < imax ; ii += 4 ) {
	    	i = ii + jj * jp + kk * kp ;

			RA = _mm256_loadu_pd(&dbl[i]);	RB = _mm256_loadu_pd(&xdbl[i]); RC1 = _mm256_add_pd(RC, _mm256_mul_pd(RA, RB));
			RA = _mm256_loadu_pd(&dbc[i]);	RB = _mm256_loadu_pd(&xdbc[i]); RC2 = _mm256_add_pd(RC1, _mm256_mul_pd(RA, RB));
			RA = _mm256_loadu_pd(&dbr[i]);	RB = _mm256_loadu_pd(&xdbr[i]); RC3 = _mm256_add_pd(RC2, _mm256_mul_pd(RA, RB));
			
			RA = _mm256_loadu_pd(&dcl[i]);	RB = _mm256_loadu_pd(&xdcl[i]); RC1 = _mm256_add_pd(RC3, _mm256_mul_pd(RA, RB));
			RA = _mm256_loadu_pd(&dcc[i]);	RB = _mm256_loadu_pd(&xdcc[i]); RC2 = _mm256_add_pd(RC1, _mm256_mul_pd(RA, RB));
			RA = _mm256_loadu_pd(&dcr[i]);	RB = _mm256_loadu_pd(&xdcr[i]); RC3 = _mm256_add_pd(RC2, _mm256_mul_pd(RA, RB));
			
			RA = _mm256_loadu_pd(&dfl[i]);	RB = _mm256_loadu_pd(&xdfl[i]); RC1 = _mm256_add_pd(RC3, _mm256_mul_pd(RA, RB));
			RA = _mm256_loadu_pd(&dfc[i]);	RB = _mm256_loadu_pd(&xdfc[i]); RC2 = _mm256_add_pd(RC1, _mm256_mul_pd(RA, RB));
			RA = _mm256_loadu_pd(&dfr[i]);	RB = _mm256_loadu_pd(&xdfr[i]); RC3 = _mm256_add_pd(RC2, _mm256_mul_pd(RA, RB));
			
			RA = _mm256_loadu_pd(&cbl[i]);	RB = _mm256_loadu_pd(&xcbl[i]); RC1 = _mm256_add_pd(RC3, _mm256_mul_pd(RA, RB));
			RA = _mm256_loadu_pd(&cbc[i]);	RB = _mm256_loadu_pd(&xcbc[i]); RC2 = _mm256_add_pd(RC1, _mm256_mul_pd(RA, RB));
			RA = _mm256_loadu_pd(&cbr[i]);	RB = _mm256_loadu_pd(&xcbr[i]); RC3 = _mm256_add_pd(RC2, _mm256_mul_pd(RA, RB));
			
			RA = _mm256_loadu_pd(&ccl[i]);	RB = _mm256_loadu_pd(&xccl[i]); RC1 = _mm256_add_pd(RC3, _mm256_mul_pd(RA, RB));
			RA = _mm256_loadu_pd(&ccc[i]);	RB = _mm256_loadu_pd(&xccc[i]); RC2 = _mm256_add_pd(RC1, _mm256_mul_pd(RA, RB));
			RA = _mm256_loadu_pd(&ccr[i]);	RB = _mm256_loadu_pd(&xccr[i]); RC3 = _mm256_add_pd(RC2, _mm256_mul_pd(RA, RB));
			
			RA = _mm256_loadu_pd(&cfl[i]);	RB = _mm256_loadu_pd(&xcfl[i]); RC1 = _mm256_add_pd(RC3, _mm256_mul_pd(RA, RB));
			RA = _mm256_loadu_pd(&cfc[i]);	RB = _mm256_loadu_pd(&xcfc[i]); RC2 = _mm256_add_pd(RC1, _mm256_mul_pd(RA, RB));
			RA = _mm256_loadu_pd(&cfr[i]);	RB = _mm256_loadu_pd(&xcfr[i]); RC3 = _mm256_add_pd(RC2, _mm256_mul_pd(RA, RB));
			
			RA = _mm256_loadu_pd(&ubl[i]);	RB = _mm256_loadu_pd(&xubl[i]); RC1 = _mm256_add_pd(RC3, _mm256_mul_pd(RA, RB));
			RA = _mm256_loadu_pd(&ubc[i]);	RB = _mm256_loadu_pd(&xubc[i]); RC2 = _mm256_add_pd(RC1, _mm256_mul_pd(RA, RB));
			RA = _mm256_loadu_pd(&ubr[i]);	RB = _mm256_loadu_pd(&xubr[i]); RC3 = _mm256_add_pd(RC2, _mm256_mul_pd(RA, RB));
			
			RA = _mm256_loadu_pd(&ucl[i]);	RB = _mm256_loadu_pd(&xucl[i]); RC1 = _mm256_add_pd(RC3, _mm256_mul_pd(RA, RB));
			RA = _mm256_loadu_pd(&ucc[i]);	RB = _mm256_loadu_pd(&xucc[i]); RC2 = _mm256_add_pd(RC1, _mm256_mul_pd(RA, RB));
			RA = _mm256_loadu_pd(&ucr[i]);	RB = _mm256_loadu_pd(&xucr[i]); RC3 = _mm256_add_pd(RC2, _mm256_mul_pd(RA, RB));
			
			RA = _mm256_loadu_pd(&ufl[i]);	RB = _mm256_loadu_pd(&xufl[i]); RC1 = _mm256_add_pd(RC3, _mm256_mul_pd(RA, RB));
			RA = _mm256_loadu_pd(&ufc[i]);	RB = _mm256_loadu_pd(&xufc[i]); RC2 = _mm256_add_pd(RC1, _mm256_mul_pd(RA, RB));
			RA = _mm256_loadu_pd(&ufr[i]);	RB = _mm256_loadu_pd(&xufr[i]); RC3 = _mm256_add_pd(RC2, _mm256_mul_pd(RA, RB));
	
			_mm256_storeu_pd(&b[i], RC3); 
	 	}
      }
   }

}

void rmatmult3_avx_fma(Domain_t *domain, RadiationData_t *rblk, double *x, double *b )
{
   char *me = "rmatmult3" ;
   int i, ii, jj, kk, tt ;
   int imin = domain->imin ;
   int imax = domain->imax ;
   int jmin = domain->jmin ;
   int jmax = domain->jmax ;
   int kmin = domain->kmin ;
   int kmax = domain->kmax ;
   int jp   = domain->jp   ;
   int kp   = domain->kp   ;
   double *dbl = rblk->dbl ;
   double *dbc = rblk->dbc ;
   double *dbr = rblk->dbr ;
   double *dcl = rblk->dcl ;
   double *dcc = rblk->dcc ;
   double *dcr = rblk->dcr ;
   double *dfl = rblk->dfl ;
   double *dfc = rblk->dfc ;
   double *dfr = rblk->dfr ;
   double *cbl = rblk->cbl ;
   double *cbc = rblk->cbc ;
   double *cbr = rblk->cbr ;
   double *ccl = rblk->ccl ;
   double *ccc = rblk->ccc ;
   double *ccr = rblk->ccr ;
   double *cfl = rblk->cfl ;
   double *cfc = rblk->cfc ;
   double *cfr = rblk->cfr ;
   double *ubl = rblk->ubl ;
   double *ubc = rblk->ubc ;
   double *ubr = rblk->ubr ;
   double *ucl = rblk->ucl ;
   double *ucc = rblk->ucc ;
   double *ucr = rblk->ucr ;
   double *ufl = rblk->ufl ;
   double *ufc = rblk->ufc ;
   double *ufr = rblk->ufr ;
   double *xdbl = x - kp - jp - 1 ;
   double *xdbc = x - kp - jp     ;
   double *xdbr = x - kp - jp + 1 ;
   double *xdcl = x - kp      - 1 ;
   double *xdcc = x - kp          ;
   double *xdcr = x - kp      + 1 ;
   double *xdfl = x - kp + jp - 1 ;
   double *xdfc = x - kp + jp     ;
   double *xdfr = x - kp + jp + 1 ;
   double *xcbl = x      - jp - 1 ;
   double *xcbc = x      - jp     ;
   double *xcbr = x      - jp + 1 ;
   double *xccl = x           - 1 ;
   double *xccc = x               ;
   double *xccr = x           + 1 ;
   double *xcfl = x      + jp - 1 ;
   double *xcfc = x      + jp     ;
   double *xcfr = x      + jp + 1 ;
   double *xubl = x + kp - jp - 1 ;
   double *xubc = x + kp - jp     ;
   double *xubr = x + kp - jp + 1 ;
   double *xucl = x + kp      - 1 ;
   double *xucc = x + kp          ;
   double *xucr = x + kp      + 1 ;
   double *xufl = x + kp + jp - 1 ;
   double *xufc = x + kp + jp     ;
   double *xufr = x + kp + jp + 1 ;

   __m256d RA, RB, RC, RC1, RC2, RC3;
   RC = _mm256_setzero_pd();

   for ( kk = kmin ; kk < kmax ; kk++ ) {
      for ( jj = jmin ; jj < jmax ; jj++ ) {
	       for ( ii = imin ; ii < imax ; ii += 4 ) {
	    	i = ii + jj * jp + kk * kp ;

			RA = _mm256_loadu_pd(&dbl[i]);	RB = _mm256_loadu_pd(&xdbl[i]); RC1 = _mm256_fmadd_pd(RA, RB, RC);
			RA = _mm256_loadu_pd(&dbc[i]);	RB = _mm256_loadu_pd(&xdbc[i]); RC2 = _mm256_fmadd_pd(RA, RB, RC1);
			RA = _mm256_loadu_pd(&dbr[i]);	RB = _mm256_loadu_pd(&xdbr[i]); RC3 = _mm256_fmadd_pd(RA, RB, RC2);
			
			RA = _mm256_loadu_pd(&dcl[i]);	RB = _mm256_loadu_pd(&xdcl[i]); RC1 = _mm256_fmadd_pd(RA, RB, RC3);
			RA = _mm256_loadu_pd(&dcc[i]);	RB = _mm256_loadu_pd(&xdcc[i]); RC2 = _mm256_fmadd_pd(RA, RB, RC1);
			RA = _mm256_loadu_pd(&dcr[i]);	RB = _mm256_loadu_pd(&xdcr[i]); RC3 = _mm256_fmadd_pd(RA, RB, RC2);
			
			RA = _mm256_loadu_pd(&dfl[i]);	RB = _mm256_loadu_pd(&xdfl[i]); RC1 = _mm256_fmadd_pd(RA, RB, RC3);
			RA = _mm256_loadu_pd(&dfc[i]);	RB = _mm256_loadu_pd(&xdfc[i]); RC2 = _mm256_fmadd_pd(RA, RB, RC1);
			RA = _mm256_loadu_pd(&dfr[i]);	RB = _mm256_loadu_pd(&xdfr[i]); RC3 = _mm256_fmadd_pd(RA, RB, RC2);
			
			RA = _mm256_loadu_pd(&cbl[i]);	RB = _mm256_loadu_pd(&xcbl[i]); RC1 = _mm256_fmadd_pd(RA, RB, RC3);
			RA = _mm256_loadu_pd(&cbc[i]);	RB = _mm256_loadu_pd(&xcbc[i]); RC2 = _mm256_fmadd_pd(RA, RB, RC1);
			RA = _mm256_loadu_pd(&cbr[i]);	RB = _mm256_loadu_pd(&xcbr[i]); RC3 = _mm256_fmadd_pd(RA, RB, RC2);
			
			RA = _mm256_loadu_pd(&ccl[i]);	RB = _mm256_loadu_pd(&xccl[i]); RC1 = _mm256_fmadd_pd(RA, RB, RC3);
			RA = _mm256_loadu_pd(&ccc[i]);	RB = _mm256_loadu_pd(&xccc[i]); RC2 = _mm256_fmadd_pd(RA, RB, RC1);
			RA = _mm256_loadu_pd(&ccr[i]);	RB = _mm256_loadu_pd(&xccr[i]); RC3 = _mm256_fmadd_pd(RA, RB, RC2);
			
			RA = _mm256_loadu_pd(&cfl[i]);	RB = _mm256_loadu_pd(&xcfl[i]); RC1 = _mm256_fmadd_pd(RA, RB, RC3);
			RA = _mm256_loadu_pd(&cfc[i]);	RB = _mm256_loadu_pd(&xcfc[i]); RC2 = _mm256_fmadd_pd(RA, RB, RC1);
			RA = _mm256_loadu_pd(&cfr[i]);	RB = _mm256_loadu_pd(&xcfr[i]); RC3 = _mm256_fmadd_pd(RA, RB, RC2);
			
			RA = _mm256_loadu_pd(&ubl[i]);	RB = _mm256_loadu_pd(&xubl[i]); RC1 = _mm256_fmadd_pd(RA, RB, RC3);
			RA = _mm256_loadu_pd(&ubc[i]);	RB = _mm256_loadu_pd(&xubc[i]); RC2 = _mm256_fmadd_pd(RA, RB, RC1);
			RA = _mm256_loadu_pd(&ubr[i]);	RB = _mm256_loadu_pd(&xubr[i]); RC3 = _mm256_fmadd_pd(RA, RB, RC2);
			
			RA = _mm256_loadu_pd(&ucl[i]);	RB = _mm256_loadu_pd(&xucl[i]); RC1 = _mm256_fmadd_pd(RA, RB, RC3);
			RA = _mm256_loadu_pd(&ucc[i]);	RB = _mm256_loadu_pd(&xucc[i]); RC2 = _mm256_fmadd_pd(RA, RB, RC1);
			RA = _mm256_loadu_pd(&ucr[i]);	RB = _mm256_loadu_pd(&xucr[i]); RC3 = _mm256_fmadd_pd(RA, RB, RC2);
			
			RA = _mm256_loadu_pd(&ufl[i]);	RB = _mm256_loadu_pd(&xufl[i]); RC1 = _mm256_fmadd_pd(RA, RB, RC3);
			RA = _mm256_loadu_pd(&ufc[i]);	RB = _mm256_loadu_pd(&xufc[i]); RC2 = _mm256_fmadd_pd(RA, RB, RC1);
			RA = _mm256_loadu_pd(&ufr[i]);	RB = _mm256_loadu_pd(&xufr[i]); RC3 = _mm256_fmadd_pd(RA, RB, RC2);
	
			_mm256_storeu_pd(&b[i], RC3); 
	 	}
      }
   }

}
