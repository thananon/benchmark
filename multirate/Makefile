CFLAGS =
MPICC = mpicc


pair: pairwise.c
	$(MPICC) $(CFLAGS) -o pairwise pairwise.c

a2a: msgrate_process.c
	$(MPICC) $(CFLAGS) -o msgrate_process msgrate_process.c

default: a2a pair

clean:
	rm -f msgrate_process pairwise
