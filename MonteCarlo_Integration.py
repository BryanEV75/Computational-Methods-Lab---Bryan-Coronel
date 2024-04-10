#Based on Monte Carlo's method
#Exercise 10.8 : Integration 
'''
I = \int_0^1 {x^{-1/2} \over{e^x +1}} dx ,

using the importance sampling formula, Eq. (10.42), with w(x) = x^-1/2, as follows.
a) Show that the probability distribution p(x) from which the sample points should 
be drawn is given by

p(x) = {1\over{2\sqrt{x}}} 

and derive a transformation formula for generating random numbers between zero and one 
from this distribution.
b) Using your formula, sample N = 1,000,000 random points and hence evaluate the integral.
You should get a value around 0.84.
'''
import matplotlib.pyplot as plt
import numpy as np

rng = np.random
N = 1000000
x_random = rng.uniform(low = 0, high = 1, size = N)

def transform_function(x0):
    """
    This function transforms a set of unfirom distributed numbers 
    to a set of numbers with a distribution of p(x)
    P(x) = 2x^-1/2

    Input:
    x0 = Normally distributed set of values

    Output:
    x = set of values with a distribution px
    """
    x = x0**2
    return x

def function(x0):
    """
    Function within the summation where g = f(x)/w(x) and
    the 2 is obtained by integrating w(x) from 0 to 1

    Input:
    x0 = Initial input of array

    Output:
    Get the function of xo to be summed 
    """
    g = 2 /( np.exp(x0) + 1 )
    return g * 2



x = transform_function(x_random) #this is our w(x) our tranformed array

function = function(x) #this is out f(x)

I = ( 1/N ) * ( np.sum( function ) )
print(f'The integral is {I:.6}.')
