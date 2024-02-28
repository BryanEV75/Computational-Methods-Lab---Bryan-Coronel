'''
Rene Bryan Coronel
Integration
Our goal is to calculate the boltzman constant
Using packages that already include those constants
Exercise 5.12
'''

import astropy.constants as cons
import astropy.units as u
import numpy as np

from scipy import integrate

#Below are mathematical constants
PI = np.pi
e = np.e
tan = np.tan
cos = np.cos

K_b = cons.k_B #This is the boltzman-constant
C = cons.c #This is the speed of light constant
Hbar = cons.hbar #This is the hbar constant

#a
#T is for temperature
#We will have to use x = tan(z) and transformt the original inetrgal
#We are showing W = alpha * T^4

constants = ( K_b**4 )/ ( 4*PI**2 * C**2 * Hbar**3 ) #T^4 should be considered seperate

def f(z):
    x = tan(z)
    return ( x**3 / (np.exp(x) - 1) ) * (1/cos(z))**2

def simpson(a,b,N):
    h = ( b - a )/ N
    s0 = 0 
    s1 = 0
    for k in range ( 1, N//2 ):
        s0 += f( a +( 2*k-1 ) * h )
    for k in range ( 1, (N//2)-1 ):
        s1 += f( a + ( 2*k*h ) )
    result = ( h/3 )*( f(a) + f(b) + 4*s0 + 2*s1 )
    return result

#b
#The answer below is not the most accurate
#We can use another method for integration
#We had to change integration parameters 
#
integral_resultant = constants * simpson(0.00000001,PI/2 - 0.00000000001,1000)

#c
print(f"{integral_resultant:.6}")
#I ran the code, our numerical value is different at the 14th significant figure
print(f"{cons.sigma_sb:.6}") #Here we are printing the Stefan-Boltzman Constant to compare to the previous integral
#We must ensure this is in the same units as the integral
#print(simpson(0.00000001,PI/2 - 0.00000000001,1000))
