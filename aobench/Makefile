#CC      = /usr/bin/cc
#CCFLAGS = -c -Wall -msse4.2 -DUSE_SSE2 -g  -O3
CC      = icc
CCFLAGS = -g -c -Wall -mavx -DUSE_SSE2  -O3
LDFLAGS = -lm -pg

SRCS    = ao.c ao_avx_soa.c ao_sse_soa.c
OBJS    = $(SRCS:.c=.o)
EXES    = $(SRCS:.c=)
OBJSd   = $(SRCS:.c=_dbg.o)
EXESd   = $(SRCS:.c=_dbg)

INCS    = 
LIBS    = 

all: $(EXES) $(EXESd)

$(OBJS): %.o: %.c
	$(CC) $(CCFLAGS) $(INCS) $< -o $@

$(EXES): %: %.o
	$(CC) $< $(LDFLAGS) -o $@

$(OBJSd): %_dbg.o: %.c
	$(CC) $(CCFLAGS) $(INCS) -DDBG $< -o $@

$(EXESd): %_dbg: %_dbg.o
	$(CC) $< $(LDFLAGS) -o $@

clean:
	rm -rf $(OBJS) $(EXES) $(OBJSd) $(EXESd) ao.ppm 
