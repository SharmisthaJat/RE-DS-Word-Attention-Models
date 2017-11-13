import os
import sys

for t in ['train','test','dev']:
    print t+"_lower"
    output = open(t+"_lower.tsv",'w')
    with open('../../Data/GIDS/'+t+'.tsv','r') as inputFile:
      for line in inputFile:
        output.write(line.strip().lower()+'\n')
    output.close()
 
