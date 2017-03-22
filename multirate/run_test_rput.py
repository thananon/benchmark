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
parser.add_option("--exec", dest="executable", default="./pairwise");
parser.add_option("-n", type="int", dest="n", default=3);
(options, args) = parser.parse_args(sys.argv)

hostname = options.hostname
mpi = options.mpi
size = options.msg_size
loop = options.iteration

#  for key, value in options.keys():
#      print key, value
#
print options


filename = "./result/"+options.executable+"_"+mpi+"_"+hostname+"_ib_"+str(size)+"byte_"+str(loop)
output = " | tee -a " + filename

for i in range(1,11):
    for iteration in range(1,options.n):
        os.system("mpirun -np 2 -host "+hostname+" -mca btl openib,self --map-by node " +options.executable+ " -s "+str(size)+"  -i "+str(loop*i)+" -Dthrds -w "+ str(256)+ output+"_sing")
    print "\n"

for i in range(1,11):
    for iteration in range(1,options.n):
        os.system("mpirun -np 2 -host "+hostname+" -mca btl openib,self --map-by node " +options.executable+ " -s "+str(size)+"  -i "+str(loop*i)+" -t 1  -w "+ str(256)+ output+"_mul")
    print "\n"


for i in range(1,11):
    for iteration in range(1,options.n):
        os.system("mpirun -np 2 -host "+hostname+" -mca btl openib,self --map-by node --bind-to socket " +options.executable+ " -s "+str(size)+"  -i "+str(loop)+" -w "+ str(256)+" -t "+str(i)+" "+ output)
    print "\n"
#      os.system("echo \"\" >> "+filename)
