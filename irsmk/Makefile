#BHEADER****************************************************************
# (c) 2006   The Regents of the University of California               *
#                                                                      *
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright       *
# notice and disclaimer.                                               *
#                                                                      *
#EHEADER****************************************************************
#CC       = xlc
#LDR      = xlc
#CC       = gcc
#LDR      = gcc
CC       = icc
LDR      = icc

#CFLAGS   = -g -c -O3 -ftree-vectorize -ftree-vectorizer-verbose=3 -ffast-math -mavx
#CFLAGS   = -g -c -O3 -fno-tree-vectorize

CFLAGS   = -g -c -O3 -xcore-avx2 -axcore-avx2 -fma
#CFLAGS   = -g -c -O3 -vec-report -mavx -msse4.2 -fma -opt-assume-safe-padding
#CFLAGS   = -g -c -O3 -vec-report -mavx2
#CFLAGS   = -c -O2 -no-vec

#LDFLAGS  = -blpdata -lm -O5
LDFLAGS  = -lm -O3 -pg

LIBS     = 
LIB_DIRS = 

PROG     = IRSmk

OBJS     = main.o rmatmult3.o utility.o

all : $(PROG) 


$(PROG) : $(OBJS)
	$(LDR)  $(LDFLAGS) -o $(PROG) $(OBJS) $(LIB_DIRS) $(LIBS)

clean :
	rm -f *.o $(PROG) core job.out *~ 


.SUFFIXES : .o  .c

#*****************************************************************
#* Rules for C object file dependencies
#*****************************************************************
.c.o :
	$(CC) $(CFLAGS) $*.c 







