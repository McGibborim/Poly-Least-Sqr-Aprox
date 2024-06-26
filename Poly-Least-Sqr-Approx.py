##################################################
# Poly-Least-Sqr-Approx.py
# by Nate McClure, Nathan Stahl, and Trey Wilkins
##################################################

import numpy as np
import matplotlib.pyplot as plt # Adds cool graph
from scipy import linalg # Access to linear algebra functions

# Main function for calculations and what not
def main():
    n = int(input("Enter the degree of the approximating polynomial P(x): ")) # Degree of the approximating polynomial P(x)
    x = np.array(input("Enter the x-values separated by spaces (3 points minimum): ").split(), float) # x-values
    y = np.array(input("Enter the y-values separated by spaces (3 points minimum): ").split(), float) # y-values (actual)
    if len(x) < 3 or len(y) < 3:
        print("Whoops, you entered less than 3 points for the x values or the y values.\nTry again.\n")
        main()
    else:
        xs, xy = get_system_of_equations(x, y, n)  # Pulls from definition get_system_of_equations
        xs = np.reshape(xs, ((n + 1), (n + 1)))  # Reshape the matrix xs to solve the system of equations
        xy = np.reshape(xy, ((n + 1), 1))
        print(xs, '\n\n', xy)
        a = np.linalg.solve(xs, xy)  # Solve the system of equations
        print('\n', a)  # Print the solution to the system of equations
        error = find_error(y, np.array(fn(x, a)))  # Pulls from definition find_error
        print("\nE =", error)
        plot(x, y, fn(x, a))  # Plot the data points and the approximating function P(x)

# Computes the values required to construct the system of equations for polynomial least-squares approximation.
def get_system_of_equations(x, y, n):
    xs = np.array([])  # Sums x-values
    xy = np.array([])  # Product of x- and y-values
    for index in range(0, (n + 1)):
        for exp in range(0, (n + 1)):
            tx = np.sum(x**(index + exp))  # \sum_{i=1}^{m}x_{i}^{j+k}
            xs = np.append(xs, tx)
        ty = np.sum(y * (x**index))  # \sum_{i=1}^{m}y_{i}x_{i}^{j}
        xy = np.append(xy, ty)
    return xs, xy

# This function calculates the error of the polynomial approximation.
def find_error(y, fn):
    return np.sum((y - fn)**2)  # E = \sum_{i=1}^{m} (y_{i} - P(x_{i}))**2

# Evaluates function.
def fn(x, a):
    px = 0
    for index in range(0, np.size(a)):
        px += (a[index] * (x**index))  # Evaluate the P(x)
    return px

# This function plots the data points (x, y) and the approximating polynomial curve P(x).
def plot(x, y, fn):
    plt.figure(figsize=(8, 6), dpi=80)
    plt.subplot(1, 1, 1)
    plt.plot(x, y, color='blue', linewidth=2.0, linestyle='-', label='y')
    plt.plot(x, fn, color='red', linewidth=3.0, linestyle='--', label='P(x)')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
