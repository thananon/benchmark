CFLAGS =
MPICC = mpicc

a2a: msgrate_process.c
	$(MPICC) $(CFLAGS) -o msgrate_process msgrate_process.c

pair: pairwise.c
	$(MPICC) $(CFLAGS) -o pairwise pairwise.c

all: a2a pair

clean:
	rm -f msgrate_process pairwise
