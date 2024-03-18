#Exercise 5.21
#DIFFERENTIATION
'''
You have two charges, of ±1 C, 10 cm apart. 
Calculate the resulting electric potential on a 1 m by 1 m square plane 
surrounding the charges and passing through them. Calculate the potential 
at 1 cm spaced points in a grid and make a visualization on the screen of 
the potential using a density plot.

Now calculate the partial derivatives of the potential with respect to 
x and y and hence find the electric field in the xy plane. Make a 
visualization of the field also. This is a little trickier than visualizing
the potential, because the electric field has both magnitude and direction. 
A visualization might use the arrow object from the visual package, drawing 
a grid of arrows with direction and length chosen to represent the field.

--> USE VECTOR PLOT
use contour plot and imshow
'''

import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as cons
import astropy.units as u

#To stay consistent we will do everything in meters
#10 cm is equivalent to 0.1 meters
#Our center point will be 0.5,0.5
#Therefore charge 1 will be at 0.45, 0.5, this will be +1
#Therefore charge 2 will be at 0.55, 0.5, this will be -1

#Dimensions of grid
x_space = np.linspace(0,1,101) #This will give us a spacing of 0.01
y_space = np.linspace(0,1,101)

#A 2D array with these dimensions 
eP_calculated = np.zeros((len(x_space),len(y_space)))
#print(eP_calculated.shape)

def electricPotential(q1, q2, x, y, space):
    '''
    q represents the point charges
    r is the distance from the origin
    Where we give an x and y array
    space is meant to represent the 2d array 
    in which the calculated pontentials will be 
    added to
    '''
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            r1 = np.sqrt((xi - 0.45)**2 + (yj - 0.5)**2 ) #This is for charge +1
            r2 = np.sqrt((xi - 0.55)**2 + (yj - 0.5)**2 ) #This is for charge -1
        
            eP1 = (q1)/ ( 4 * np.pi * cons.eps0.value * r1) #This is for charge +1
            eP2 = ( q2 )/ ( 4 * np.pi * cons.eps0.value * r2) #This is for charge -1
            space[i,j] = eP1 + eP2 #


electricPotential(1, -1, x_space, y_space, eP_calculated)

#Create a density plot of the potential
        
plt.figure("Density Plot")
plt.imshow(eP_calculated.T, extent = (0,1 , 0, 1), origin ='lower', cmap='cool')
plt.colorbar(label='Electric Potential (V)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.title('Electric Potential')
plt.show()

plt.figure("Contour Plot")
cs = plt.contour( x_space, y_space, eP_calculated.T, levels = 15, extent = ( 0, 1, 0, 1 ) )
plt.clabel(cs, fontsize = 10, fmt = '%1.2f')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.title('Electric Potential')
plt.show()

##################################################################################################

def gradient_calculated(f):
    """
    We will base our differetiation on the central difference theorem
    dfdx = ( f(x + h/2, y) - f(x - h/2, y ) ) / h
    dfdy = ( f(x, y + h/2) - f(x, y - h/2) ) / h
    """
    h = 0.01 #This our step size
    
    dfdx = np.zeros_like(f)#This will create arrays of the same size as the grid
    dfdy = np.zeros_like(f) 

    for i in range ( 1, f.shape[1] - 1 ): #We are skipping the first index and the last one to avoid any errors
        dfdx[i , :] = ( f[ i + 1, : ] - f[i - 1, : ] )/ ( 2 * h ) # We are indexing the grid and adding 1 to follow the next index and dividing by 0.01
        #Note that in our grid we have already previously defined each step size to be 0.01, therefore by indexing by 1 we are stepping forward by h
        

    for j in range ( 1, f.shape[0] - 1 ):
        dfdy[: , j] = ( f[ : , j + 1 ] - f[ : , j - 1 ] )/ ( 2 * h )

    return dfdx, dfdy

dvdx, dvdy = gradient_calculated(eP_calculated)
print(dvdx.shape)
Ex = -dvdx
Ey = -dvdy

#To visualize we need direction and magnitude 
E = np.sqrt ( Ex**2 + Ey**2 ) #This is the magnitude, which will represent length of vector
#direction = np.arctan(Ey/Ex) #


plt.figure("EM fields by numeric approach")
plt.title('Electromagnetic Field') 
plt.streamplot(x_space, y_space, Ex.T, Ey.T, density = 1.4)
plt.plot(0.45,0.50,'or') #This is charge of 1
plt.plot(0.55,0.50,'ob') #This is charge of -1
plt.grid()
plt.show()

#################################################################
#I used the function below to compare results 
dVdx, dVdy = np.gradient(eP_calculated, x_space, y_space)
print(dVdx.shape) #We will use this to verify our differentiation below 
plt.figure("EM field by built in function")
plt.title('Electromagnetic Field') 
plt.streamplot(x_space, y_space, -dVdx.T, -dVdy.T, density = 1.4)
plt.plot(0.45,0.50,'or') #This is charge of 1
plt.plot(0.55,0.50,'ob') #This is charge of -1
plt.grid()
plt.show()
#plt.quiver(x_space, y_space, Ex.T, Ey.T)