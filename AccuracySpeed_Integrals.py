#Exercise 4.4 : Calculating Integrals
'''
a)We want to evaluate the integral and get a
resultant of pi/2, with N = 100 slices
b)We will increase the value of N and check how
long it takes the program to run
'''
import numpy as np
import argparse 
from tqdm import tqdm #Will calculate time needed to run program

parser = argparse.ArgumentParser(description = "This program calcuates a function's intergral using Riemann Sum")
parser.add_argument('-N',type=float, default=10 , action='store'
                   ,help=" The N argument determines the number of slices in the Riemann Sum")
args = parser.parse_args()

def intergrate (a, b, N): #Integrates from a to b
    h = 2/N #Used to calculate the width of each slice
    y = 0 #This is a starting array that will be summed over
    x = np.arange(0,N+1,1)
    #print(x)
    for k in tqdm(range(len(x))):
        x[k] = -1 + h * k 
        y = y + ( np.sqrt( 1 - x[k]**2 ) ) * h
    
    return y

print("Intergral = ",intergrate(-1,1,args.N))
print("True value = ",np.pi/2)

