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
filename = "result_ptoq_"+str(size)+"byte_"+str(loop)

output = " | tee -a " + filename

#  for i in range(1,11):
#      for iteration in range(1,4):
#          os.system("mpirun -np "+str(i+1)+" -host arc08 -mca btl vader,self --map-by core ./msgrate_process -s "+str(size)+" -m "+ str(i) +" -n 1 -i "+str(loop)+"  -Dthrds "+ output)
#      print "\n"
#      os.system("echo \"\" >> "+filename)
#
#
#  for i in range(1,11):
#      for iteration in range(1,4):
#          os.system("mpirun -np "+str(i+1)+" -host arc08 -mca btl vader,self --map-by core ./msgrate_process -s "+str(size)+" -m "+ str(i) +" -n 1 -i "+str(loop)+"  "+ output)
#      print "\n"
#      os.system("echo \"\" >> "+filename)

for i in range(1,11):
    for iteration in range(1,4):
        os.system("mpirun -np 2 -host arc08 -mca btl vader,self --map-by core ./msgrate_process -s "+str(size)+" -n 1 -m 1 -i "+str(loop)+" -y " + str(i) + output)
    print "\n"
    os.system("echo \"\" >> "+filename)

#  os.system("echo \"\" >> "+filename)
#  os.system("echo \"\" >> "+filename)
#  os.system("echo \"\" >> "+filename)
#
#  for i in range(1,11):
#      for iteration in range(1,4):
#          os.system("mpirun -np "+str(i+1)+" -host arc08 -mca btl vader,self --map-by core ./msgrate_process -s "+str(size)+" -n "+ str(i) +" -m 1 -i "+str(loop)+"  -Dthrds "+ output)
#      print "\n"
#      os.system("echo \"\" >> "+filename)
#
#
#  for i in range(1,11):
#      for iteration in range(1,4):
#          os.system("mpirun -np "+str(i+1)+" -host arc08 -mca btl vader,self --map-by core ./msgrate_process -s "+str(size)+" -n "+ str(i) +" -m 1 -i "+str(loop)+"  "+ output)
#      print "\n"
#      os.system("echo \"\" >> "+filename)
#
#  for i in range(1,11):
#      for iteration in range(1,4):
#          os.system("mpirun -np 2 -host arc08 -mca btl vader,self --map-by core ./msgrate_process -s "+str(size)+" -n 1 -m 1 -i "+str(loop)+" -x " + str(i) + output)
#      print "\n"
#      os.system("echo \"\" >> "+filename)
#
#
#  #  for i in range(1,11):
#      for iteration in range(1,4):
#          os.system("mpirun -np "+str(i*2)+" -host arc08 -mca btl vader,self --map-by core ./pairwise -s "+str(size)+" -n "+ str(i) +" -m " +str(i) +" -i "+str(loop)+ output)
#      print "\n"
#      os.system("echo \"\" >> "+filename)
#
#  for i in range(1,11):
#      for iteration in range(1,4):
#          os.system("mpirun -np 2 -host arc08 -mca btl vader,self --map-by socket --bind-to socket ./pairwise -s "+str(size)+"  -x "+ str(i) +" -y " +str(i) + " -n 1 -m 1 -i "+str(loop)+ " "+ output)
#      print "\n"
#      os.system("echo \"\" >> "+filename)
