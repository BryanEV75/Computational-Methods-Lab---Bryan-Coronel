#Based on Monte Carlo's method
#Exercise 10.3 : Brownian Motion
'''
Brownian motion is the motion of a particle, such as a smoke or dust particle,
in a gas, as it is buffeted by random collisions with gas molecules. Make a 
simple computer simulation of such a particle in two dimensions as follows. 
The particle is confined to a square grid or lattice L x L squares on a side, 
so that its position can be represented by two integers i, j = 0 . . . L - 1. 
It starts in the middle of the grid. On each step of the simulation, choose a 
random direction—up, down, left, or right—and move the particle one step in that 
direction. This process is called a random walk. The particle is not allowed to 
move outside the limits of the lattice—if it tries to do so, choose a new random 
direction to move in.

Write a program to perform a million steps of this process on a lattice with 
L = 101 and make an animation on the screen of the position of the particle. 
(We choose an odd length for the side of the square so that there is one lattice 
site exactly in the center.)
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation 

rng = np.random #This is our random number generator
L = 101 

#i,j = np.arange(0, L-1), np.arange(0, L-1)
#print(i.shape, j.shape)


#We will perform 1 million steps

#Our starting point can be the center
#50, 50 

#We could think as N as seconds or frames passing
def brownian_motion(N):
    """
    This function uses a random number generator

    Input:
    N = Number of steps 
    """

    x,y = np.zeros(N), np.arange(N)
    for t in range (1,N):
        #This is our starting point
        x[0] = 50
        y[0] = 50

        result = rng.randint(low = 1, high = 5, size = 1)

        if result == 1: #This is for right
            x[t] = x[t - 1] + 1
            y[t] = y[t - 1]
            if x[t] == 100:
                x[t] = 99

        if result == 2: #This is for left
            x[t] = x[t - 1] - 1
            y[t] = y[t - 1]
            if x[t] == 0:
                x[t] = 1
        
        if result == 3: #This is for up
            x[t] = x[t - 1] 
            y[t] = y[t - 1] + 1
            if y[t] == 100:
                y[t] = 99

        if result == 4: #This is for down
            x[t] = x[t - 1] 
            y[t] = y[t - 1] - 1
            if y[t] == 0:
                y[t] = 1
    return x,y


x,y = brownian_motion(1000000)


#This is so we can visualize what is happening 
#plt.plot(x,y)
#plt.plot(50,50,marker = '*')
#plt.xlim(0,100)
#plt.ylim(0,100)
#plt.xlabel('x')
#plt.ylabel('y')
#plt.show()

#Now we have to animate the motion 
fig  = plt.figure( figsize=(8,8) )
ax = plt.axes(xlim = (0,100), ylim = (0,100))
particle = plt.Circle( (50,50), radius = 1, facecolor = 'red')

def init():
    particle.center = (50,50)
    ax.add_patch(particle)
    return particle

def animate(i):
    for patch in ax.patches:
        patch.remove()
    
    particle.center = ( x[i], y[i])
    ax.add_patch(particle)

    trail_length = 100
    for j in range(max(0, i - trail_length),i):
        trail_particle = plt.Circle((x[j],y[j]), radius = 0.5, fc = 'grey',
                                    alpha = 0.1)
        ax.add_patch(trail_particle)
    return particle,

anim = animation.FuncAnimation(fig, animate,frames=len(x), init_func= init)
#anim.save("Brownian_Motion.mp4")
plt.show()
#I know I can add an input for N 



