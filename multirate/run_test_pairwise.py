#!usr/bin/python


import sys
import os
import subprocess
from optparse import OptionParser
hostname = "arc06,arc07"
mpi = "master"
size = 1024;
loop = 1000;


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


filename = mpi+"_"+hostname+"_ib_"+str(size)+"byte_"+str(loop)
output = " | tee -a " + filename

for i in range(1,11):
    for iteration in range(1,4):
        os.system("mpirun -np "+str(i*2)+" --hostfile ./hostfile  -host "+hostname+" -mca btl openib,self --map-by node --bind-to core ./pairwise -s "+str(size)+" -i "+str(loop)+"  -Dthrds "+ output)
    print "\n"
    os.system("echo \"\" >> "+filename)

for i in range(1,11):
    for iteration in range(1,4):
        os.system("mpirun -np "+str(i*2)+" --hostfile ./hostfile  -host "+hostname+" -mca btl openib,self --map-by node --bind-to core ./pairwise -s "+str(size)+" -i "+str(loop)+ output)
    print "\n"
    os.system("echo \"\" >> "+filename)

#  for i in range(1,11):
#      for iteration in range(1,2):
#          os.system("mpirun -np 2 -host arc08 -mca btl vader,self --map-by core --bind-to core ./pairwise -s "+str(size)+"  -x "+ str(i) +" -y " +str(i) + " -n 1 -m 1 -i "+str(loop)+ " "+ output)
#      print "\n"
#      os.system("echo \"\" >> "+filename)

#  for i in range(1,11):
#      for iteration in range(1,3):
#          os.system("mpirun -np 2 -host "+hostname+" -mca btl openib,self --map-by node --bind-to core ./pairwise -s "+str(size)+"  -x "+ str(i) +" -y " +str(i) + " -n 1 -m 1 -i "+str(loop)+ " "+ output)
#      print "\n"
#      os.system("echo \"\" >> "+filename)
