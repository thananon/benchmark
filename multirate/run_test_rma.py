#!usr/bin/python


import sys
import os
import subprocess
from optparse import OptionParser
hostname = "arc06,arc07"
mpi = "master"
size = 1024;
loop = 100;


def run_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE)
    return iter(p.stdout.readline, b'')

parser = OptionParser()

parser.add_option("--host", dest="hostname", default=hostname,
        help="specify the host for mpirun.")

parser.add_option("--mpi", dest="mpi", default=mpi,
        help="specify the branch for mpirun.")

parser.add_option("--size", type="int", dest="msg_size",  default=size)
parser.add_option("--iter", type="int", dest="iteration", default=loop)
(options, args) = parser.parse_args(sys.argv)

hostname = options.hostname
mpi = options.mpi
size = options.msg_size
loop = options.iteration

#  for key, value in options.keys():
#      print key, value
#
print options


filename = "./result/rma/"+mpi+"_ib_"+str(size)+"byte_"+str(loop)+"_rma"
output = " | tee -a " + filename

#  for i in range(1,11):
#      for iteration in range(1,6):
#          os.system("mpirun -np 2 -host "+hostname+" -mca btl openib,self --map-by node --bind-to none ./pairwise_rma -s "+str(size)+"  -i "+str(loop*i)+" -Dthrds -w "+ str(256)+ output+"_sing")
#      print "\n"
#
#  for i in range(1,11):
#      for iteration in range(1,6):
#          os.system("mpirun -np 2 -host "+hostname+" -mca btl openib,self --map-by node --bind-to none ./pairwise_rma -s "+str(size)+"  -i "+str(loop*i)+" -t 1  -w "+ str(256)+ output+"_mul")
#      print "\n"
#

for i in range(1,11):
    for iteration in range(1,6):
        os.system("mpirun -np 2 -mca btl openib,self --map-by node --bind-to socket ./pairwise_rma -s "+str(size)+"  -i "+str(loop)+" -w "+ str(128)+" -t "+str(i)+" "+ output)
    print "\n"
    os.system("echo \"\" >> "+filename)
