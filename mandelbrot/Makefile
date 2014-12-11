CC = icc
#CC = g++
#CFLAGS = -g -c -O3 -funroll-loops -ftree-vectorizer-verbose=1 -flax-vector-conversions -ffast-math -mavx -march=native -fomit-frame-pointer -D_GNU_SOURCE -DGCC
#CFLAGS = -g -c -O3 -I./ #-fno-tree-vectorize -fno-tree-slp-vectorize 

CFLAGS = -g -c -O3 -vec-report -xcore-avx2 -D_GNU_SOURCE -DICC

LDFLAGS = -pg -lm -lrt

all: mandelbrot_dbl mandelbrot_dbl_avx 

mandelbrot_dbl: mandelbrot_dbl.o
	$(CC) mandelbrot_dbl.o -o mandelbrot_dbl $(LDFLAGS)

mandelbrot_dbl.o: mandelbrot_dbl.cpp
	$(CC) $(CFLAGS) $< -o $@

mandelbrot_dbl_avx: mandelbrot_dbl_avx.o
	$(CC) mandelbrot_dbl_avx.o -o mandelbrot_dbl_avx $(LDFLAGS)

mandelbrot_dbl_avx.o: mandelbrot_dbl_avx.cpp
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf *.o mandelbrot_dbl 
