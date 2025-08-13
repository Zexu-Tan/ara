#!bin/python

import re
import os

 

datadir = "vector_traces"
kernel_undertest = os.environ["kernel_undertest"]

trace = f"{datadir}/trace_hart_0.log"

dump = f"{datadir}/{kernel_undertest}.dump"



print("===================VECTORFLOW===================")

print(f"Kernel: {kernel_undertest}")

print("================================================")

 

#inststr = r"\s+([0-9a-f]{8}):"

vinststr = r"([0-9a-f]{8}):\s+[0-9a-f\s]+(v.*)"

imem = {}

 

with open(dump, "r") as f:

    function = ""

    for line in f:

        if match := re.match(vinststr, line):

            adr = int(match.group(1), 16)

            function = match.group(2)

            imem[adr] = function

 

new_trace = ""

 

with open(trace, "r") as f:

    for line in f:
        
        if "vector arith" in line:
            adr = int(line.split()[3], 16)
            vinst = imem[adr]
            new_trace += line.replace("vector arith", vinst)
            
        elif "VLx" in line or "VSx" in line:
            adr = int(line.split()[3], 16)
            vinst = imem[adr]
            new_trace += line.replace(line.split()[6], vinst)

        else:
            new_trace += line
            continue
            

        
        
            
        

 

fo = "trace_hart_0_plus_vinstr.txt"

# Open file in write mode

 

with open(fo, "w") as file:

    # Write the variable content into the file

    file.write(new_trace)
