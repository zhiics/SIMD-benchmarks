cc = icc
#cc = gcc

CFLAGS = -g -c -O3 -DICC -D_OPENMP -vec-report -xcore-avx2 

#LDFLAGS = -pg -lm -lrt -fopenmp
LDFLAGS = -pg -lm -lrt 

all: spectralnorm_dbl spectralnorm_dbl_sse spectralnorm_dbl_avx spectralnorm_dbl_avx2 

spectralnorm_dbl: spectralnorm_dbl.o
	$(cc) spectralnorm_dbl.o -o spectralnorm_dbl $(LDFLAGS)

spectralnorm_dbl_sse: spectralnorm_dbl_sse.o
	$(cc) spectralnorm_dbl_sse.o -o spectralnorm_dbl_sse $(LDFLAGS)

spectralnorm_dbl_avx: spectralnorm_dbl_avx.o
	$(cc) spectralnorm_dbl_avx.o -o spectralnorm_dbl_avx $(LDFLAGS)

spectralnorm_dbl_avx2: spectralnorm_dbl_avx2.o
	$(cc) spectralnorm_dbl_avx2.o -o spectralnorm_dbl_avx2 $(LDFLAGS)

spectralnorm_dbl.o: spectralnorm_dbl.c
	$(cc) $(CFLAGS) -inline-forceinline -no-prec-div $< -o $@ $(LDFLAGS)

spectralnorm_dbl_sse.o: spectralnorm_dbl_sse.c
	$(cc) $(CFLAGS) $< -o $@ $(LDFLAGS)

spectralnorm_dbl_avx.o: spectralnorm_dbl_avx.c
	$(cc) $(CFLAGS) $< -o $@ $(LDFLAGS)

spectralnorm_dbl_avx2.o: spectralnorm_dbl_avx2.c
	$(cc) $(CFLAGS) -march=core-avx2 $< -o $@ $(LDFLAGS)

clean:
	rm -rf *.o sepctralnorm_dbl
