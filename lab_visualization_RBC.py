"""
Rene Bryan Coronel 
Exercise 3.2
Use argparse and plot all of them at the same time 
or plot them individually
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

pi = np.pi
e = np.e
cos = np.cos
sin = np.sin

#a
#theta has to be between 0 and 2pi
#Deltoid Curve

theta_0= np.linspace(0,2*pi,1000)
def deltoid_curve(theta):
    x = 2*cos(theta)+cos(2*theta)
    y = 2*sin(theta)-sin(2*theta)
    return x,y

#deltoid_curve(theta_0)

#b
#r = f(theta) for some function, in this case its cos and sin 
#we then convert r and theta to cartesian coordinates
#x = rcos(theta) and y = rsin(theta)
#Galilean Spiral

theta_1 = np.linspace(0,10*pi,1000)
def galilean_spiral(theta):
    r = theta**2
    x = r*cos(theta)
    y = r*sin(theta)
    return x,y

#galilean_spiral(theta_1)

#c
#Here we have a different function for r 
#Fey's function

theta_2 = np.linspace(0,24*pi,1000)
def feys_function(theta):
    r = e**cos(theta)-2*cos(4*theta)+(sin(theta/12)**5)
    x = r*cos(theta)
    y = r*sin(theta)
    return x,y

#feys_function(theta_2)

def all_graphs(a,b,c):
    x0,y0 = deltoid_curve(a)
    x1,y1 = galilean_spiral(b)
    x2,y2 = feys_function(c)


    axs = plt.subplots(1,3, figsize=(10,5), num="All Functions Plotted")
    axs[0].plot(x0,y0)
    axs[0].set_title("Deltoid Curve")
    axs[0].set_ylabel("y")
    axs[1].plot(x1,y1)
    axs[1].set_title("Galilean Spiral")
    axs[1].set_xlabel("x")
    axs[2].plot(x2,y2)
    axs[2].set_title("Fey's Function")
    plt.show()
#all_graphs(theta_0,theta_1,theta_2)

parser = argparse.ArgumentParser(description="We want to either print all of them with argument 'all' or print each\
    inidivdual graph with their corresponding title 'deltoid','galilean','fey'.")
parser.add_argument('string', choices=['deltoid', 'fey', 'Galilean','all'],
                    help='Options are: deltoid, fey, Galilean,all')
args = parser.parse_args()
chosen_string = args.string.lower() #This will convert string to lowercase so input doesn't matter
print(f'Chosen string: {chosen_string}')

#All functions are defined beforehand 
if chosen_string == 'deltoid':
    x,y = deltoid_curve(theta_0)
    plt.figure("Deltoid Curve")
    plt.title("Deltoid Curve")
    plt.plot(x,y)
    plt.show()
elif chosen_string=='galilean':
    x,y = galilean_spiral(theta_1)
    plt.figure("Galilean Spiral")
    plt.title("Galilean Spiral")
    plt.plot(x,y)
    plt.show()
elif chosen_string == 'fey':
    x,y = feys_function(theta_2)
    plt.figure("Fey's Function")
    plt.title("Fey's Function")
    plt.plot(x,y)
    plt.show()
elif chosen_string == 'all':
    all_graphs(theta_0,theta_1,theta_2)
else:
    print("Not a valid input")