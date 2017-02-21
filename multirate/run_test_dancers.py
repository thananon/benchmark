#!usr/bin/python


import sys
import os
import subprocess

def run_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE)
    return iter(p.stdout.readline, b'')



size = 1024;
loop = 1000;
host = "d17_18"
filename = "ib_fix_"+host+"_ib_"+str(size)+"byte_"+str(loop)
output = " | tee -a " + filename

#  for i in range(1,11):
#      for iteration in range(1,4):
#          os.system("mpirun -np 2 -host arc08 -mca btl vader,self --map-by core ./pairwise -s "+str(size)+" -n 1 -m 1 -s 64 -i "+str(i*loop)+"  -Dthrds "+ output)
#      print "\n"
#      os.system("echo \"\\n\" >> "+filename)
#
#  for i in range(1,11):
#      for iteration in range(1,4):
#          os.system("mpirun -np 2 -host arc08 -mca btl vader,self --map-by core ./pairwise -s "+str(size)+" -n 1 -m 1 -s 64 -i "+str(i*loop)+"  "+ output)
#      print "\n"
#      os.system("echo \"\\n\" >> "+filename)

#  for i in range(1,11):
#      for iteration in range(1,4):
#          os.system("mpirun -np "+str(i*2)+" -host arc08 -mca btl vader,self --map-by core ./pairwise -s "+str(size)+" -n "+ str(i) +" -m " +str(i) + " -i "+str(loop)+"  -Dthrds "+ output)
#      print "\n"
#      os.system("echo \"\" >> "+filename)
#
#
#  for i in range(1,11):
#      for iteration in range(1,4):
#          os.system("mpirun -np "+str(i*2)+" -host arc08 -mca btl vader,self --map-by core ./pairwise -s "+str(size)+" -n "+ str(i) +" -m " +str(i) +" -i "+str(loop)+ output)
#      print "\n"
#      os.system("echo \"\" >> "+filename)
#
#  for i in range(1,11):
#      for iteration in range(1,2):
#          os.system("mpirun -np 2 -host arc08 -mca btl vader,self --map-by core --bind-to core ./pairwise -s "+str(size)+"  -x "+ str(i) +" -y " +str(i) + " -n 1 -m 1 -i "+str(loop)+ " "+ output)
#      print "\n"
#      os.system("echo \"\" >> "+filename)

for i in range(1,11):
    for iteration in range(1,6):
        os.system("mpirun -np 2 -host d17,d18 -mca btl openib,self --map-by node --bind-to socket ./pairwise -s "+str(size)+"  -x "+ str(i) +" -y " +str(i) + " -n 1 -m 1 -i "+str(loop)+ " "+ output)
    print "\n"
    os.system("echo \"\" >> "+filename)
