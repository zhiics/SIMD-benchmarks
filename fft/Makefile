#CC = gcc
CC = icc
#CFLAGS = -g -c -O3 -ftree-vectorizer-verbose=1 -ffast-math -mavx2 -march=native 
CFLAGS = -g -c -O3 -mavx -vec-report
DFLAGS = -lm -pg -lrt

all: fft compare dcompare  

# fft implementation  
fft:driver.o fft.o 
	$(CC) driver.o fft.o -o fft $(DFLAGS)

fft_dup:driver.o fft_dup.o 
	$(CC) driver.o fft_dup.o -o fft_dup $(DFLAGS)

driver.o:driver.c
	$(CC) $(CFLAGS) driver.c -o driver.o -lm

# utilities for comparing two files with floating point numbers 
compare: compare.c 
	$(CC) compare.c -o compare -lm

dcompare: dcompare.c 
	$(CC) dcompare.c -o dcompare -lm

clean: 
	rm -f *.o dcompare compare fft fft_dup core 

run:
