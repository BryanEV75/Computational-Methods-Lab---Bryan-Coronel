#Exercise 4.3
#Calculating Derivative 
import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()

def f(x):
    '''
    This is the function that will calculate our value
    for f(x) from the problem
    '''
    return x*(x-1)

#a
delta = 1E-2
x = 1
df = ( f(x + delta) - f(x) ) / delta
print(df)
'''
We get 1.01...
Mathematically we should get 1
We do not get the same answer because mathematically we know 1 is an integer 
but within the computer, 1 has more values depedning on the signifcant figures
'''

#We can turn the derivative into a function

def derivatives(delta,x=1):
    df = ( f(x + delta) - f(x) ) / delta
    return df

#b 
deltas = np.array([1E-2,1E-4, 1E-6, 1E-8, 1E-10, 1E-12, 1E-14])
derivates = derivatives(deltas)
'''
We can plot the array of deltas on the x-axis using log to limit spread
We can also multiply by -1 so that we have a positive axis
We can see how within our graph some of the derivatives calculated stray 
from the true derivative 
'''

plt.figure(figsize=(10,8),facecolor='w')
plt.scatter (np.log(deltas), derivates,color='black',marker="*",alpha=0.9)
plt.axhline(y=1, linewidth = 0.2,label = "True derivative")
plt.xlabel("log10($\delta$)")
plt.ylabel("Derivative Values (df)")
plt.legend()
plt.show()

#Calculating/ and displaying the time it takes to run the script
end = time.time()
print (f"This script took {end-start:.2} sec to run")
