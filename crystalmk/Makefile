#BHEADER****************************************************************
# (c) 2007   The Regents of the University of California               *
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

#-mfpmath=sse #force to use sse instead of FPU

#CFLAGS   = -c -qlanglvl=stdc99 -O3 -qalias=allp -qtune=auto -qarch=auto -qcache=auto  
#CFLAGS   = -c -O3 -fno-tree-vectorize -msse4.2
CFLAGS   = -c -O3 -xcore-avx2
#CFLAGS   = -c -O3 -vec-report
#CFLAGS   = -c -O3 -no-vec

#LDFLAGS  = -blpdata -lm
LDFLAGS  = -lm -pg

LIBS     = 
LIB_DIRS = 

PROG     = CrystalMk 

OBJS     = main.o  init.o SPEdriver.o Crystal_div.o Crystal_pow.o Crystal_Cholesky.o 

all : $(PROG) 

$(PROG) : $(OBJS)
	$(LDR)  $(LDFLAGS) -o $(PROG) $(OBJS) $(LIB_DIRS) $(LIBS) -lm

clean :
	rm -f *.o $(PROG) core job.out *~ 


.SUFFIXES : .o  .c

#*****************************************************************
#* Rules for C object file dependencies
#*****************************************************************
.c.o :
	$(CC) $(CFLAGS) $*.c 







