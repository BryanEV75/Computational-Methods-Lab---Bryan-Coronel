#Basic imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import astropy.constants as cons

#Matrix imports 
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs

#Particle properties
hbar = 1 # We could use cons.hbar.value but value of 1 is standard in analytics
mass = 1 # cons.m_e.value m also gets standardized to 1 
#This gives us just -1/2

#Grid boxed
N = 100
L = 1
X,Y = np.meshgrid( np.linspace(0, L, N, dtype = float), np.linspace(0, L, N, dtype = float))

#First potential being tested
def potential(x,y):
    return 0*X
V = potential(X,Y)

#Matrix manipulation

#Matrix with ones
diag = np.ones(N)

#Referring back to our D matrix, this creates the 1,-2,1 pattern we see across the diagonal
diags = np.array([diag, -2*diag, diag]) 

#Refer back to our matrix D
#Essentially, the matrix starts with -2 , 1, 0
#In the next line we have 1, -2, 1
#The code below moves the matrix with ones to be pushed down, the matrix with
#-2*diag to be where we start and the other diag matrix to be push to the right
D = sparse.spdiags(diags, np.array([-1, 0, 1]), N, N)

#Here we simply use a function already found, and multiply our constans
#Note again then hbar and mass are denoted to be 1 for simplification
T = -(hbar / 2 * mass) * sparse.kronsum(D,D)

#Here we are reshaping the potential to be one column x N*2
#The zero simply centers it in the matrix
U = sparse.diags(V.reshape(N**2), (0))

#This is simply the hamiltonian 
H = T + U

#Looks for smallest vectors and values where k is the energy states
#This is extracting it from that 1D array where each row is placed into a combined column 
eigenvalues, eigenvectors = eigs(H, k = 10, which = "SM")

#n are reffered to as the state levels that we are used to
def eigenvector(n):
    return eigenvectors.T[n].reshape(N,N)

#Plot for potential
plt.figure( figsize = (5,5))
plt.title("Plot for potential(V)")
plt.xlabel("X")
plt.ylabel("Y")
plt.contourf(X,Y,V,100)
plt.colorbar()
plt.show()

#This section plots the probability density
for w in range(6):
    plt.figure( figsize=(5,5))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Plot of the probability denisty for the {w} level state ")
    plt.contourf(X,Y,eigenvector(w)**2,25)
    plt.colorbar()
    plt.show()

#This section is for the energy levels
alpha = eigenvalues[0]/2 #We are using the ground state
E_div_alpha = eigenvalues/alpha # normalizing the enegry levels
_ = np.arange(0, len(eigenvalues), 1)
plt.figure(figsize=(5,5))
plt.scatter(_, E_div_alpha)

#Theorethical lines for energy levels
[plt.axhline(nx**2+ny**2,color = 'b') for nx in range (1,5) for ny in range (1,5)]
plt.show()

###########################################################################################
#For circular well
def potential_circular(x,y,r):
    V = np.zeros_like(x)
    outside_circle = np.sqrt((x-L/2)**2 + (y-L/2)**2) > r
    V[outside_circle] = 1e10
    return V
V_c = potential_circular(X,Y,0.25)

U_c = sparse.diags(V_c.reshape(N**2),(0))
H_c = T + U_c

eigenvalues_c, eigenvectors_c = eigs(H_c, k = 10, which = "SM")

def eigenvector_c(n):
    return eigenvectors_c.T[n].reshape(N,N)

#Plot for potential
plt.figure( figsize = (5,5))
plt.title("Plot for potential(V)")
plt.contourf(X,Y,V_c,100)
plt.colorbar()
plt.show()

#Plot of probability denisty
for w in range(6):
    plt.figure( figsize=(5,5))
    plt.title(f"Plot of the probability denisty for the {w} level state ")
    plt.contourf(X,Y,eigenvector_c(w)**2,20)
    plt.show()

#Plot of energy levels
alpha_c = eigenvalues_c[0]/2
E_div_alpha_c = eigenvalues_c/alpha_c
_ = np.arange(0, len(eigenvalues_c), 1)
plt.figure(figsize=(5,5))
plt.scatter(_, E_div_alpha_c)
plt.show()
