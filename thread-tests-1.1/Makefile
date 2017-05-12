#
# Copyright (c) 2016      Research Organization for Information Science
#                         and Technology (RIST). All rights reserved.
#
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

CC   = mpicc
FLAGS = -pthread

C_BINS   = $(shell ls *.c | sed 's/..$$//')

all: $(C_BINS)

%: %.c
	$(CC) $(FLAGS) $^ -o $@

%.o: %.c
	$(CC) $(FLAGS) -c $<

clean:
	rm -f $(C_BINS) *.o
