CFLAGS =
MPICC = mpicc


pair: pairwise.c
	$(MPICC) $(CFLAGS) -o pairwise pairwise.c 

a2a: msgrate_process.c
	$(MPICC) $(CFLAGS) -o msgrate_process msgrate_process.c

rma: pairwise_rma.c
	$(MPICC) $(CFLAGS) -o pairwise_rma pairwise_rma.c

rput: pairwise_rput.c
	$(MPICC) $(CFLAGS) -o pairwise_rput pairwise_rput.c

all: a2a pair rma rput

clean:
	rm -f msgrate_process pairwise pairwise_rma pairwise_rput
