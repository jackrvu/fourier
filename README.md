### Fourier Transform Definition
The Fourier Transform of a function \( f(x) \) is defined as:
```math
\hat{f}(k) = \int_{-\infty}^{\infty} f(x) e^{-i k x} \, dx
```
where:
'''math
- \( f(x) \) is the original function in the spatial domain,
- \( \hat{f}(k) \) is the transformed function in the frequency domain,
- \( k \) is the frequency variable.
'''
The inverse Fourier Transform is given by:
```math
 f(x) = \frac{1}{2\pi} \int_{-\infty}^{\infty} \hat{f}(k) e^{i k x} \, dk
```

### Application to PDEs
For many PDEs, applying the Fourier Transform reduces the problem to solving algebraic equations in the frequency domain. This simplifies the problem and allows us to solve for the solution in terms of its Fourier components.
@@ -66,16 +66,16 @@ For many PDEs, applying the Fourier Transform reduces the problem to solving alg

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
@@ -103,3 +103,19 @@ u = np.zeros((len(x), len(t)))
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
