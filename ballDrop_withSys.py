import sys
import numpy as np
sqrt = np.sqrt
#We want to test for a 100 meter tower
print(sys.argv)
#sys.argv is a list
h = float(sys.argv[1]) 
t = sqrt(2*h/9.81)
print(f"It takes {t:.2} seconds for the ball to fall from {h} meters.")