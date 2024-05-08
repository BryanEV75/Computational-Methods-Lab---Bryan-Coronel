'''
Exercise 8.7: Trajectory with air resistance

Many elementary mechanics problems deal with the physics of objects moving or flying 
through the air, but they almost always ignore friction and air resistance to make the 
equations solvable. If we’re using a computer, however, we don’t need solvable equations.

Consider, for instance, a spherical cannonball shot from a cannon standing on level 
ground. The air resistance on a moving sphere is a force in the opposite direction to the 
motion with magnitude

F = \frac{1}{2} \pi R^2\rho C v^2, 

where R is the sphere’s radius, ρ is the density of air, v is the velocity, and C is the 
so-called coefficient of drag (a property of the shape of the moving object, in this case
 a sphere).

Starting from Newton’s second law, F = ma, show that the equations of motion for the 
position (x, y) of the cannonball are

\ddot{x} = - {\pi R^2\rho C\over2m}\, \dot{x}\sqrt{\dot{x}^2+\dot{y}^2},
\ddot{y} = - g - {\pi R^2\rho C\over2m}\, \dot{y}\sqrt{\dot{x}^2+\dot{y}^2},

where m is the mass of the cannonball, g is the acceleration due to gravity, and 
\dot{x} and \ddot{x} are the first and second derivatives of x with respect to time.

Change these two second-order equations into four first-order equations using the 
methods you have learned, then write a program that solves the equations for a cannonball 
of mass 1 kg and radius 8 cm, shot at 30º to the horizontal with initial velocity 100 ms−1.
The density of air is ρ = 1.22 kg m−3 and the coefficient of drag for a sphere is C = 0.47.
Make a plot of the trajectory of the cannonball (i.e., a graph of y as a function of x).

When one ignores air resistance, the distance traveled by a projectile does not depend on 
the mass of the projectile. In real life, however, mass certainly does make a difference. 
Use your program to estimate the total distance traveled (over horizontal ground) by the 
cannonball above, and then experiment with the program to determine whether the cannonball 
travels further if it is heavier or lighter. You could, for instance, plot a series of 
trajectories for cannonballs of different masses, or you could make a graph of distance 
traveled as a function of mass. Describe briefly what you discover.
'''

import numpy as np
import plotly.express as px #Does things easily, seaborn
import plotly.graph_objects as go #Builds plot from level up, matplotlib
import astropy.constants as cons
import matplotlib.pyplot as plt

g = cons.g0.value
theta = np.radians(30) #Radians (Originally in degrees)
m = 1 #kg
R = 0.08 #meters
V0 = 100 #meters/seconds (Intial Velocity)
p = 1.22 #kg/meters^3
C = 0.47 #Coefficient of drag
x0,y0 = 0,0 #Initial positions in meters (Starting from origin)

def f(r,t):
    """
    Computes the change of the x and y velocity and
    the change in their acceleration

    Input: 
    array: initial array 

    Output 
    dVxdt,dVydt,dxdt,dydt: At some given point 

    """
    Vx = r[0]
    Vy = r[1]
    x = r[2]
    y = r[3]
    V = np.sqrt( Vx**2 + Vy**2)
    dxdt = Vx
    dydt = Vy
    dVxdt = - (np.pi * R**2 * p * C * Vx * V) / (2*m)
    dVydt = - g - (np.pi * R**2 * p * C * Vy * V) / (2*m)
    return np.array([dVxdt,dVydt,dxdt,dydt],float)

# Time interval and step size
t0, tf = 0, 7
h = 0.1

# Create arrays for t and r values
t_points = np.arange(t0, tf, h)

# Solve equations of motion using the fourth-order Runge-Kutta method
r = np.array([V0*np.cos(theta), V0*np.sin(theta), x0, y0], float)

# Runge-Kutta integration function
def Runge_Kuta(function, r0, t_points):
    """
    Computes the fourth order Runge Kunta

    Inputs: 
    func : A function that describes the change in the two variables
    h : Step size
    r0 : Starting array
    t_poimnts: An array of the time for each passing point

    Output: 
    r : array with the calculated variables 

    """
    r = np.zeros ((len(t_points), len(r0)))
    r[0] = r0
    for i in range(1, len(t_points)):
        k1 = h * f(r[i - 1], t_points[i - 1])
        k2 = h * f(r[i - 1] + 0.5 * k1, t_points[i - 1] + 0.5 * h)
        k3 = h * f(r[i - 1] + 0.5 * k2, t_points[i - 1] + 0.5 * h)
        k4 = h * f(r[i - 1] + k3, t_points[i - 1] + h)
        r[i] = r[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return r

results = Runge_Kuta(f, r, t_points)
x_points = results[:,2]
y_points = results[:,3]


# Plot trajectory
plt.plot(x_points, y_points)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Cannonball Trajectory')
plt.show()

#Plot using plotly express
label_dict = dict(x = "Distance (meters)", y = "Height (meters)")
fig = px.line(x = x_points, y = y_points, title = "Cannonbal Trajectory", labels = label_dict)
fig.update
fig.show()

'''
#Plot using plotly graph objects
FIG = go.Figure()
FIG.add_trace(go.Scatter( x = x_points, y = y_points, mode = "lines"))
FIG.update_layout(title = "Cannonball Trajectory", xaxis_title = "Height (meters)",
                  yaxis_title = "Height (meters)", hovermode = "closest")
FIG.show()
'''
#######################################################################
#WE NEED TO REPEAT THE ABOVE BUT WITH mass varying
masses = np.array([0.5,1,5,10,15])

label_dict = dict(x="Distance (meters)", y="Height (meters)")
fig = px.line(title="Cannonball Trajectory", labels=label_dict)

t2_points = np.arange(0, 10, h)

# Iterate over each mass value
for m in masses: 
    results = Runge_Kuta(f, r, t2_points)

    x_points = results[:, 2]
    y_points = results[:, 3]

    # Add a trace for this mass value
    fig.add_scatter(x=x_points, y=y_points, mode='lines', name=f"Mass {m}")

# Show the plot
fig.show()
print("We can see that as the mass of the object, the distance traveled increases.")