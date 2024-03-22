#Exercise 7.1
'''
Write Python programs to calculate the coefficients in the 
discrete Fourier transforms of the following periodic functions 
sampled at N = 1000 evenly spaced points, and make plots of their
amplitudes:

a) A single cycle of a square-wave with amplitude 1
b) The sawtooth wave yn = n
c) The modulated sine wave yn = sin(πn/N) sin(20πn/N)
'''
#Put docstring for each of your function 

import numpy as np
import matplotlib.pyplot as plt

N = 1000 #This is the spacing we will use for most of our code
n1 = np.linspace(0,1,N)
n2 = np.linspace(0,1,N)
n3 = np.linspace(0,2*np.pi,N)

#a - Asingle cycle of a square-wave with ampltiude 1

def square_wave(x0):
    """
    This function computes the values of a square wave between 0 and 1 given an array or value.
    Returns an array or value.
    -----------------------------

    Inputs
    x0: input value or array

    ---------------------------
    Outputs

    y: value or array

    """
    y = np.zeros_like(x0)
    for i , x in enumerate(x0):
        if x < 0.5:
            y[i] = 1
        else:
            y[i] = -1
    return y

#plt.figure()
#plt.plot(n1,square_wave(n1))
#plt.show()

#b - The sawtooth wave yn = n

def sawtooth_wave(x0):
    """
    This function computes the values of a sawtooth wave between 0 and 1 given an array or value.
    Returns an array or value.
    -----------------------------

    Inputs
    x0: input value or array

    ---------------------------
    Outputs

    y: value or array

    """
    y = np.zeros_like(x0)
    for i, x in enumerate(x0):
        if x < 0.25:
            y[i] = x / 0.25
        elif x >= 0.25 and x <= 0.75:
            y[i] = 1 - ( (x - 0.25) / 0.25 )
        else:
            y[i] = -1 + ( (x - 0.75 ) / 0.25)
    return y

#plt.figure()
#plt.plot(n2,sawtooth_wave(n2))
#plt.show()

#c - The modulated sine wave yn = sin(πn/N) sin(20πn/N)
#I think this function may be wrong or N being large is what makes such a wide spread
def modulated_sine_wave(x0):
    N = 1
    """
    This function computes the values of a modulated sine wave between 0 and 2π given an array or value.
    Returns an array or value.
    -----------------------------

    Inputs
    x0: input value or array

    ---------------------------
    Outputs

    y: value or array

    """
    y = np.zeros_like(x0)
    for i,x in enumerate(x0):
        y[i] = np.sin( (np.pi * x ) / N ) * np.sin( (20 * np.pi * x) / N )
    return y
 
#plt.figure()
#plt.plot(n3,modulated_sine_wave(n3,N))
#plt.show()

#Discrete Fourier transformation and inverse fourier transformation below 
def dft(y0):
    """
    This function computes the discrete fourier transformation of given values. 
    -----------------------------

    Inputs
    y0: input array that will be transformed

    ---------------------------
    Outputs

    c: value or array after transformation

    """
    N = len(y0)
    c = np.zeros(N//2 +1, complex)
    for k in range(N//2 + 1):
        for n in range(N):
            c[k] += y0[n] * np.exp(-2j * np.pi * k * n / N )
    return c #These are ck, 'fourier coefficients'

def inverse_dft(c):
    """
    This function computes the inverse fourier transfomration usinhg a given fourier coefficients.
    -----------------------------

    Inputs
    c: input value or array of Fourier coefficients 

    ---------------------------
    Outputs

    yk: Returns inverted values

    """
    yk = np.zeros(N, complex)

    for n in range(N):
        for k, ck in enumerate(c):
            yk[n] += ck * np.exp(2j * k * np.pi * n / N )
    yk = yk/N
    return yk #This is the real fourier coefficients


###BElOW IS A PLOT OF ALL FUNCTIONS TOGETHER 

fig, ax = plt.subplots(1,3,figsize=(18,6))

ax[0].plot(n1, square_wave(n1))
ax[0].set_title("Square Wave")

ax[1].plot(n2,sawtooth_wave(n2))
ax[1].set_title("Sawtooth Wave")

ax[2].plot(n3,modulated_sine_wave(n3))
ax[2].set_title("Modulated Sine Wave")
fig.suptitle("Functions Prior to Fourier Transform", size=15)
plt.show()

#We set up our fourier transformations for each function 
t_SqrW = dft( square_wave(n1) )#Square wave
t_SawW = dft( sawtooth_wave(n2) )#Sawtooth wave
t_SinW = dft( modulated_sine_wave(n3) )#Modulated sine wave

'''
dft_SqrW calculatures the discrete fourier trasnformation of the square wave equation which includes complex parts
np.conjusgate(dft_SqrW) calculates the conjugate of those transformed values
dft_SqrW * np.conjugate(dft_SqrW) gives us the coefficents with any complex parts
We need the conjugate in order to be able to plot
Note that we also started with a real function 
'''

coef= np.arange(0,501,1)

fig, ax = plt.subplots(1,3,figsize=(18,6))

ax[0].plot(coef, t_SqrW * np.conjugate(t_SqrW))
ax[0].set_title("Square Wave")
ax[0].set_xlim(0,20)

ax[1].plot(coef, t_SawW * np.conjugate(t_SawW))
ax[1].set_title("Sawtooth Wave")
ax[1].set_xlim(0,20)

ax[2].plot(coef, t_SinW * np.conjugate(t_SinW))
ax[2].set_title("Modulated Sine Wave")
ax[2].set_xlim(0,20)
fig.suptitle("Amplitudes", size=15)
plt.show()