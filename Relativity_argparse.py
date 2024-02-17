'''
Rene Bryan Coronel exercise 2.4
A spaceship travels from Earth in a straight line at relativistic speed v to another planet x 
light years away. Write a program to ask the user for the value of x and the speed v as a 
fraction of the speed of light c, then print out the time in years that the spaceship takes 
to reach its destination (a) in the rest frame of an observer on Earth and (b) as perceived 
by a passenger on board the ship. Use your program to calculate the answers for a planet 10 
light years away with v = 0.99c.
use arseparse to take distance x and speed v
'''

#Our unit will be in terms of light years
import argparse
import numpy as np 
sqrt = np.sqrt
c = 3*1e8

parser = argparse.ArgumentParser(description="We want to (a) calculate time it takes for arrival from the rest frame on earth and\
                                 (b) the arrival time from the ship's perspective ")
parser.add_argument("distance", type=float, help="Enter how far the planet is in lightyears")
parser.add_argument("velocity", type=float, help="Enter the relative velocity as a fraction of the speed of light")

args = parser.parse_args()

print(f"The planet is {args.distance} lightyears away.")
print(f"Relative velocity is {args.velocity}c.")

gamma = 1/(sqrt(1-(args.velocity)**2))

#We want to find the time it takes for the rest frame on earth
to = args.distance/(args.velocity)
print(f"It takes {to:.3} light years to reach the other planet from the rest frame on earth perspective")

#We then want to find the time it takes for a passanger within the ship
t = to/gamma
print(f"It takes {t:.3} light years to reach the other planet from the perspective of a passenger on the ship.")
