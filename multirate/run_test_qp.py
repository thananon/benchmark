#!usr/bin/python


import sys
import os
from subprocess import call

for i in range(1,6):
    for iteration in range(1,2):
        os.system("mpirun -np "+str(i*2)+" -host arc08 -mca btl vader,self --map-by core --bind-to socket ./pairwise -s 64 -n "+ str(i) +" -m " +str(i) + " -i 1000 -Dthrds | tee -a result_pairwise_single")

#  for i in range(1,6):
#      for iteration in range(1,2):
#          os.system("mpirun -np 2 -host arc08 -mca btl vader,self --map-by core --bind-to socket ./pairwise -s 64 -x "+ str(i) +" -y " +str(i) + " -n 1 -m 1 -i 1000 | tee -a result_pairwise")
