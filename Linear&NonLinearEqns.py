#Exercise 6.14
#1 
'''
1. We want to write the follow three equations and plot them along
side each other from E = 0 to E = 20 eV

2. Write a second program to calculate the values of the first six 
energy levels in electron volts to an accuracy of 0.001 eV using binary search.
'''

import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as c
import astropy.units as u

#Constants 
m_e = c.m_e.value
hbar = c.hbar.value
w = 1e-9 
E = 1

eV = u.electronvolt
eV_to_joules = eV.to(u.J) 
V_in_joules= 20 * eV_to_joules #This is the highest energy level

#Original space
E = np.linspace(0,20,1000)
E_in_joules = E * eV_to_joules

#Equations
y01 = np.tan( np.sqrt( ( w**2 * m_e * E_in_joules ) / ( 2 * hbar**2 )) )
y02 = np.sqrt( ( V_in_joules - E_in_joules ) / ( E_in_joules ) )
y03 = -1 * np.sqrt( ( E_in_joules )  / ( V_in_joules - E_in_joules ))

#Below is our initial graph 
'''
y01[:-1][np.diff(y01) < 0] = np.NaN # gets rid of lines to infinity from tangent values
plt.plot(E, y01, color = 'k', lw = 1.4, alpha = 0.9, label = 'y1')
plt.plot(E, y02, alpha = 0.7, label  = 'y2')
plt.plot(E, y03, alpha = 0.7, label = 'y3')
plt.axhline(y = 0, color = 'grey', linestyle = '--') 
plt.xlabel('E (eV)')
plt.xlim(0,20)
plt.ylim(-4,4)
plt.legend(loc = "best")
plt.show()
'''

#To make this easier we will make the y a function of E,

def y1(E):
     E_in_joules = E * eV_to_joules
     y1 = np.tan( np.sqrt( ( w**2 * m_e * E_in_joules ) / ( 2 * hbar**2 )) )
     return y1

def y2(E):
    E_in_joules = E * eV_to_joules
    y2 = np.sqrt( ( V_in_joules - E_in_joules ) / ( E_in_joules ) )
    return y2

def y3(E):
    E_in_joules = E * eV_to_joules
    y3 = -1 * np.sqrt( ( E_in_joules )  / ( V_in_joules - E_in_joules ))
    return y3


#SUBTRACT EACH FUNCTION FROM EACHOTHER 
#With the function below we can put y2 and y3
#We can get y1 - y2
def y1_diff_y2(E):
    y = y1(E) - y2(E)
    return y

#We can get y1 - y2
def y1_diff_y3(E):
    y = y1(E) - y3(E)
    return y

#We have to define a bisection function

def bisection_method(x1,x2,function,eps= 0.001):
    while np.abs(x1 - x2) > eps : #This determines accuracy 

        xp = (x1 + x2)/2 #This is the midpoint

        if function(xp) == 0:
            return xp
        elif np.sign(function(x1)) == np.sign(function(xp)):
            x1 = xp
        else:
            x2 = xp

    return xp

#We deal with y1-y2 here
point1 = bisection_method(2.82,2.94, y1_diff_y2)
point2 = bisection_method(7.74,8.06, y1_diff_y2)
point3 = bisection_method(14.57,15.36, y1_diff_y2)

#We deal with y1 - y3
point4 = bisection_method(1.17,1.33, y1_diff_y3)
point5 = bisection_method(4.92,5.20, y1_diff_y3)
point6 = bisection_method(11.05,11.41, y1_diff_y3)

#These are our plots
fig,ax =plt.subplots(1,3, figsize=(14,8))
y01[:-1][np.diff(y01) < 0] = np.NaN
ax[0].plot(E,y01,label='y1')
ax[0].plot(E,y02,label='y2')
ax[0].plot(E,y03,label='y3')
ax[0].axhline(color='red',linestyle='--' )
ax[0].set_ylim(-5,5)
ax[0].set_xlim(0,20)
ax[0].set_xlabel('E(eV)')
ax[0].set_title('y1, y2, y3')
ax[0].legend(loc = "lower right")

ax[1].plot(E,y01 - y02)
ax[1].scatter(point1, 0, marker = 'x', c = 'r', label = 'bisection point', s = 50)
ax[1].scatter(point2, 0, marker = 'x', c = 'r', s = 50)
ax[1].scatter(point3, 0, marker = 'x', c = 'r', s = 50)
ax[1].axhline(color='red',linestyle='--')
ax[1].set_xlabel('E(eV)')
ax[1].set_title("y1 - y2")
ax[1].set_ylim(-1,1)
ax[1].set_xlim(0,20)

ax[2].plot(E,y01-y03)
ax[2].scatter(point4, 0, marker = 'x', c = 'r', label = 'bisection point', s = 50)
ax[2].scatter(point5, 0, marker = 'x', c = 'r', s = 50)
ax[2].scatter(point6, 0, marker = 'x', c = 'r', s = 50)
ax[2].axhline(color='red',linestyle='--')
ax[2].set_xlabel('E(eV)')
ax[2].set_title("y1 - y3")
ax[2].set_ylim(-1,1)
ax[2].set_xlim(0,20)

plt.show()
print("The first six energy levels can be visaulzied through the first graph.")
print(f"The first six energy levels are calculated to be, {point1:.2}eV, {point2:.2}eV, {point3:.2}eV, {point4:.2}eV, {point5:.2}eV, and {point6:.2}eV.")