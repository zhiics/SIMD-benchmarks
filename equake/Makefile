###############################################################################
# SPECtools2000
#   Version 0.3
#   This makefile should NOT be changed other than to customize for a 
#   particular benchmark.  Users may overide variables by placing them
#   in the Makefile.spec file.  Or by using the "runspec" program to 
#   build the Makefile.spec file for them.
###############################################################################
#	BENCHMARK	-- standard definitions for this benchmark

#VERSION=AVX

CC=icc
CFLAGSAVX=-g -c -O3 -vec-report -xcore-avx2 -DAVX -no-prec-div -opt-assume-safe-padding
CFLAGSSSE=-g -c -O3 -vec-report -msse4.2 -DSSE
BASICCFLAGS=-g -c -O3 -vec-report

#ifdef VERSION
#	ifeq "$(VERSION)" "AVX"
#		CFLAGS += -mavx -DAVX
#	endif
#	ifeq "$(VERSION)" "SSE"
#		CFLAGS += -msse4.2 -DSSE
#	endif
#endif

all:quake_dbl quake_sse quake_avx quake_avx_fma 

quake_dbl:quake_dbl.o
	$(CC) quake_dbl.o -o quake_dbl -lrt

quake_dbl.o:quake.c
	$(CC) $(BASICCFLAGS) $< -o $@

quake_sse:quake_sse.o
	$(CC) quake_sse.o -o quake_sse -lrt

quake_sse.o:quake_sse.c
	$(CC) $(CFLAGSSSE) $< -o $@

quake_avx:quake_avx.o
	$(CC) quake_avx.o -o quake_avx -lrt

quake_avx.o:quake_avx.c
	$(CC) $(CFLAGSAVX) $< -o $@

quake_avx_fma:quake_avx_fma.o
	$(CC) quake_avx_fma.o -o quake_avx_fma -lrt

quake_avx_fma.o:quake_avx_fma.c
	$(CC) $(CFLAGSAVX) $< -o $@

clean:
	rm *.o quake quake_avx quake_sse quake_avx_fma
