import numpy as np,matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
#import seaborn as sns
from scipy.stats import norm

bad_words = ['#']
with open('cl2scale.post') as oldfile, open('cl2scale.txt', 'w') as newfile:
    for line in oldfile:
        if not any(bad_word in line for bad_word in bad_words):
            newfile.write(line)
            

with open('cl09.post') as oldfile, open('cl09.txt', 'w') as newfile:
    for line in oldfile:
        if not any(bad_word in line for bad_word in bad_words):
            newfile.write(line)
            
with open('cl20.post') as oldfile, open('cl20.txt', 'w') as newfile:
    for line in oldfile:
        if not any(bad_word in line for bad_word in bad_words):
            newfile.write(line)
            
with open('clZcolumn.post') as oldfile, open('clZcolumn.txt', 'w') as newfile:
    for line in oldfile:
        if not any(bad_word in line for bad_word in bad_words):
            newfile.write(line)

with open('cl_ratio.post') as oldfile, open('cl_ratio.txt', 'w') as newfile:
    for line in oldfile:
        if not any(bad_word in line for bad_word in bad_words):
            newfile.write(line)
            
with open('cl_spherevary.post') as oldfile, open('cl_spherevary.txt', 'w') as newfile:
    for line in oldfile:
        if not any(bad_word in line for bad_word in bad_words):
            newfile.write(line)


with open('clgg.post') as oldfile, open('clgg.txt', 'w') as newfile:
    for line in oldfile:
        if not any(bad_word in line for bad_word in bad_words):
            newfile.write(line)      

with open('clirreg.post') as oldfile, open('clirreg.txt', 'w') as newfile:
    for line in oldfile:
        if not any(bad_word in line for bad_word in bad_words):
            newfile.write(line)    
            
            
with open('cl63lam.post') as oldfile, open('cl63lam.txt', 'w') as newfile:
    for line in oldfile:
        if not any(bad_word in line for bad_word in bad_words):
            newfile.write(line)   

with open('clparacol.post') as oldfile, open('clparacol.txt', 'w') as newfile:
    for line in oldfile:
        if not any(bad_word in line for bad_word in bad_words):
            newfile.write(line)   
            
with open('clvoro.post') as oldfile, open('clvoro.txt', 'w') as newfile:
    for line in oldfile:
        if not any(bad_word in line for bad_word in bad_words):
            newfile.write(line)   

