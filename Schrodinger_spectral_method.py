'''
Exercise 9.9
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.fft import rfft, irfft


L = 1e-8
M = 9.109e-31 #kg (Mass of electron)
x0 = L/2
omega = 1e-10
kappa = 5e10 

def WaveFunction_zero(x,t):
    #Recall j is complex number
    result = np.exp(-(x-x0)**2 / (2 * omega**2)) * np.exp(1j * kappa * x )
    return result

N = 1000
x= np.linspace(0,L,N)

y= WaveFunction_zero(x,0)

#Used to visualize the wave function at t = 0 
plt.figure(figsize = (8,6))
plt.title("Wave Function at t = 0")
plt.plot(x,y)
plt.show()

#Now we turn the Wave function at t = 0 into a Fourier transformation
#to get bk

def dst(y):
    N = len(y)
    y2 = np.empty (2*N,float)
    y2[0] = y2[N] = 0.0
    y2[1:N] = y[1:]
    y2[:N:-1] = -y[1:]
    a = -np.imag(rfft(y2))[:N]
    b = np.real(rfft(y2)[:N])
    a[0] = 0.0 #imaginary part
    b[0] = 0.0 #real part
    return a,b

nk, ak = dst(y)
#ak is the real part
#nk is the complex part

def time_evolution(real,img,t=1e-16): 
    h= 1e-16
    evo_coefs= np.zeros(N)

    for i in range(N) :

        evo_coefs = np.sin( np.pi * kappa * x[i] ) * (ak * np.cos((np.pi**2) * h * 
            ( kappa**2 ) * t / ( 2 * M * ( L**2)))  -  
            nk * np.sin( (np.pi**2) * h * ( kappa**2 ) * t/ ( 2* M* (L**2))) )
        
        evo_coefs += evo_coefs
     
    evo_coefs = evo_coefs/N

    return evo_coefs

evo_coefs = time_evolution(ak, nk)

#Now we do the inverse of the transformation 
def idst(a):
    N = len(a)
    c = np.empty(N+1,complex)
    c[0] = c[N] = 0.0
    c[1:N] = -1j*a[1:]
    y = irfft(c)[:N]
    y[0] = 0.0

    return y

evo_inversed = idst(evo_coefs)

plt.figure(figsize=(8,6))
plt.plot(x,evo_inversed)
plt.title("Evolved Wave Function at t=1e-16")
plt.xlabel("x (m)")
plt.ylabel('y (m)')
plt.show()