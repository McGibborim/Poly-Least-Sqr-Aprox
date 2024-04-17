import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

def main():
    n = int(input("Enter the degree of the approximating polynomial P(x): ")) # Degree of the approximating polynomial P(x)
    x = np.array(input("Enter the x-values separated by spaces: ").split(), float) # x-values
    y = np.array(input("Enter the y-values separated by spaces: ").split(), float) # y-values (actual)
    xs, xy = get_system_of_equations(x, y, n)  # Get the system of equations
    xs = np.reshape(xs, ((n + 1), (n + 1)))  # Reshape the matrix xs to solve the system of equations
    xy = np.reshape(xy, ((n + 1), 1))
    print(xs, '\n\n', xy)
    a = np.linalg.solve(xs, xy)  # Solve the system of equations
    print('\n', a)  # Print the solution to the system of equations
    error = find_error(y, np.array(fn(x, a)))  # Determine the error of P(x)
    print("\nE =", error)
    plot(x, y, fn(x, a))  # Plot the data points and the approximating function P(x)

def get_system_of_equations(x, y, n):
    xs = np.array([])  # Summation of x-values
    xy = np.array([])  # Product of x- and y-values
    for index in range(0, (n + 1)):
        for exp in range(0, (n + 1)):
            tx = np.sum(x**(index + exp))  # \sum_{i=1}^{m}x_{i}^{j+k}
            xs = np.append(xs, tx)
        ty = np.sum(y * (x**index))  # \sum_{i=1}^{m}y_{i}x_{i}^{j}
        xy = np.append(xy, ty)
    return xs, xy

def find_error(y, fn):
    return np.sum((y - fn)**2)  # E = \sum_{i=1}^{m} (y_{i} - P(x_{i}))**2

def fn(x, a):
    px = 0
    for index in range(0, np.size(a)):
        px += (a[index] * (x**index))  # Evaluate the P(x)
    return px

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