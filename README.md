# Fourier Transformations in PDEs

## Overview

This project explores the application of Fourier Transforms in solving Partial Differential Equations (PDEs). The Fourier Transform is a mathematical tool that decomposes a function into its constituent frequencies, making it an essential technique for solving PDEs, particularly in fields like signal processing, heat conduction, and wave propagation.

### Purpose
The goal of this project is to demonstrate the power of Fourier Transforms in solving specific classes of PDEs by:
- Applying Fourier Transforms to simplify the equation.
- Analyzing the results in both the frequency and spatial domains.
- Solving practical PDE problems using Fourier techniques.

## Table of Contents
1. [Introduction](#introduction)
2. [Mathematical Background](#mathematical-background)
3. [Methods](#methods)
4. [PDE Examples Solved](#pde-examples-solved)
5. [Implementation](#implementation)
6. [Results](#results)
7. [Conclusion](#conclusion)
8. [References](#references)

## Introduction

Partial Differential Equations (PDEs) describe a wide range of physical phenomena, including heat diffusion, wave propagation, and fluid dynamics. Fourier Transforms offer a powerful method for solving these equations, particularly in problems with periodic boundary conditions or in problems where the solution can be expressed as a sum of sinusoidal components.

This project aims to:
- Introduce the concept of Fourier Transforms and their role in solving PDEs.
- Explore a few classic PDE examples and apply the Fourier Transform technique to find their solutions.
- Show how Fourier Transforms can simplify complex PDEs and facilitate solutions in practical scenarios.

## Mathematical Background

### Fourier Transform Definition
The Fourier Transform of a function \( f(x) \) is defined as:
```math
\hat{f}(k) = \int_{-\infty}^{\infty} f(x) e^{-i k x} \, dx
```
where:
- \( f(x) \) is the original function in the spatial domain,
- \( \hat{f}(k) \) is the transformed function in the frequency domain,
- \( k \) is the frequency variable.

The inverse Fourier Transform is given by:
```math
 f(x) = \frac{1}{2\pi} \int_{-\infty}^{\infty} \hat{f}(k) e^{i k x} \, dk
```

### Application to PDEs
For many PDEs, applying the Fourier Transform reduces the problem to solving algebraic equations in the frequency domain. This simplifies the problem and allows us to solve for the solution in terms of its Fourier components.

## Methods

### Fourier Transform in Solving PDEs
1. **Problem Setup**: The PDE is expressed in the form of a linear differential equation.
2. **Fourier Transform**: We apply the Fourier Transform to convert the spatial domain problem into a frequency domain problem.
3. **Solve in Frequency Domain**: The transformed PDE is usually easier to solve algebraically or numerically.
4. **Inverse Fourier Transform**: After solving the equation in the frequency domain, we apply the inverse Fourier Transform to obtain the solution in the spatial domain.

### Types of PDEs Solved
- **Heat Equation**: Using Fourier Transform to solve the one-dimensional heat equation.
- **Wave Equation**: Solving the wave equation with periodic boundary conditions.
- **Laplace's Equation**: Applying Fourier series for solutions in rectangular domains.

## PDE Examples Solved

### Example 1: One-dimensional Heat Equation
The heat equation in one dimension is given by:
```math
\frac{\partial u(x,t)}{\partial t} = \alpha \frac{\partial^2 u(x,t)}{\partial x^2}
```
Using the Fourier Transform, we can reduce this to an algebraic equation in the frequency domain.

### Example 2: Wave Equation
The wave equation is given by:
```math
\frac{\partial^2 u(x,t)}{\partial t^2} = c^2 \frac{\partial^2 u(x,t)}{\partial x^2}
```
We solve it using Fourier Transforms for a fixed initial condition and periodic boundary conditions.

## Implementation

The implementation is done in Python, using the `numpy` library for numerical computations and the `matplotlib` library for visualization. The code performs the following steps:

1. Discretize the spatial domain.
2. Apply the Fourier Transform to the PDE.
3. Solve the transformed equation in the frequency domain.
4. Compute the inverse Fourier Transform to get the solution in the spatial domain.

```python
import numpy as np
import matplotlib.pyplot as plt

# Example: Solving heat equation with Fourier Transform
# Define constants and initial conditions

# Discretization
x = np.linspace(0, 10, 100)
t = np.linspace(0, 2, 50)
u = np.zeros((len(x), len(t)))

# Initial condition (for example)
u[:, 0] = np.sin(np.pi * x)

# Solve using Fourier Transform...
```

## Results

The results demonstrate how Fourier Transforms simplify solving PDEs by converting them into algebraic equations. Visualizations show the solutions evolving over time, verifying theoretical predictions.

## Conclusion

Fourier Transforms provide a powerful method for solving PDEs by decomposing functions into their frequency components. This technique is particularly useful for problems with periodic boundary conditions and can be extended to more complex PDEs in higher dimensions.

## References

1. J. Fourier, "Théorie analytique de la chaleur," 1822.
2. E. Kreyszig, "Advanced Engineering Mathematics," 10th Edition.
3. R. Haberman, "Applied Partial Differential Equations," 5th Edition.

