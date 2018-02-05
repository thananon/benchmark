#!usr/bin/python


import sys
import os
import subprocess
from optparse import OptionParser
mpi = "master"
size = 4000;
loop = 200;


def run_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE)
    return iter(p.stdout.readline, b'')

parser = OptionParser()

parser.add_option("--mpi", dest="mpi", default=mpi,
        help="specify the branch for mpirun.")

parser.add_option("--size", type="int", dest="msg_size",  default=size)
parser.add_option("--iter", type="int", dest="iteration", default=loop)
(options, args) = parser.parse_args(sys.argv)

mpi = options.mpi
size = options.msg_size
loop = options.iteration

#  for key, value in options.keys():
#      print key, value
#
print options


filename = "./result/multi_module_t2t/"+mpi+"_ib_"+str(size)+"byte_"+str(loop)+"_p2p"
output = " | tee -a " + filename

#  for i in range(1,11):
#      for iteration in range(1,4):
#          os.system("mpirun -np 2 -host "+hostname+" -mca btl openib,self --map-by node --bind-to none ./pairwise -s "+str(size)+"  -i "+str(loop*i)+" -Dthrds -w "+ str(256)+ output+"_sing")
#      print "\n"

#  for i in range(1,11):
#      for iteration in range(1,4):
#          os.system("mpirun -np 2 -host "+hostname+" -mca btl openib,self --map-by node --bind-to none ./pairwise -s "+str(size)+"  -i "+str(loop*i)+" -t 1  -w "+ str(256)+ output+"_mul")
#      print "\n"


#  for i in range(1,11):
#      for iteration in range(1,4):
#          os.system("mpirun -np 2 -mca btl openib,self --map-by node --bind-to socket ./pairwise -s "+str(size)+"  -i "+str(loop)+" -w "+ str(64)+" -t "+str(i)+" "+ output)
#      print "\n"
#      os.system("echo \"\" >> "+filename)

for i in range(1,11):
    for iteration in range(1,11):
        os.system("mpirun -np " + str(2) + " -mca pml ucx --map-by node --bind-to socket ./pairwise -s "+str(size)+"  -i "+str(loop)+" -w "+ str(128)+" -t " + str(i) +"  "+ output)
        #  os.system("mpirun -np " + str(2) + " -mca btl_openib_btls_per_lid 1 -mca btl_openib_receive_queues \"S,128,1024,256,16:S,2048,1024,1008,256:S,4096,1024,1008,128:S,65536,1024,1008,128\" -mca btl_openib_warn_default_gid_prefix 0 -mca btl openib,self --map-by node --bind-to socket ./pairwise -s "+str(size)+"  -i "+str(loop)+" -w "+ str(128)+" -t " + str(i) +"  "+ output)
    print "\n"
    os.system("echo \"\" >> "+filename)

#  for i in range(1,11):
#      for iteration in range(1,10):
#          os.system("mpirun -np " + str(2) + " -rr -genv I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=disable -genv I_MPI_PIN_DOMAIN=core ./pairwise -s "+str(size)+"  -i "+str(loop)+" -w "+ str(128)+" -t " +str(i)+ "  " + output)
#      print "\n"
#      os.system("echo \"\" >> "+filename)

