# Week 7

Exercises: https://drive.google.com/file/d/18EKZWK1el56vdh5k2EfKIt_BphZDd8-R/view?usp=sharing
Literature: ESL Chapters: 4.5, 12.5, 12.2, 12.3.1
Slides: https://drive.google.com/file/d/1pttnS-N43txk-19RI7OwgnQPMS6Zg9h7/view?usp=sharing
Subjects: Basis expansion, Support Vector Machine and convex optimization

# Optimal Separating Hyperplane

In a binary classification setting we will assume that the two classes are completely separable. There would exists many hyperplanes which would separate the two classes, but what would be the optimal?

The distance between the two classes are to be maximised. We’ll introduce the margin C which is defined to be the distance from the decision boundary to the closest points of each class.

![Untitled](Week%207%20acec4d8daa834447bd9d2bb36af00d39/Untitled.png)

First we’ll need to define how we get a distance from a point to a plane.

![Untitled](Week%207%20acec4d8daa834447bd9d2bb36af00d39/Untitled%201.png)

This can then be made into a maximisation problem

$$
\arg\max_{\beta, \beta_0} C
$$

such that 

$$
y_i \frac{x_i\beta + \beta_0}{||\beta||} \geq C \quad \forall i
$$

This maximisation problem is turned into a minimisation problem by letting $C$ be equal to $\frac{1}{||\beta||}$ making us minimise the norm of $\beta$, 

$$
\begin{align*}
\arg\min_{\beta, \beta_0} \frac12 ||\beta||^2 \text{s.t.} \\

y_i(x_i\beta + \beta_0) \geq 1 \quad \forall i
\end{align*}
$$

This problem is quadratic because of the squared $\beta$ and thus a nonlinear problem with linear constraints.

Something that would be nice to see is if we could have a single coefficient for each observation instead of each dimension. This would be a good idea for high-dimensional problems with few observations. Also, a non-linear separation between classes. This can be achieved using Lagrange multipliers.

# Constrained optimisation crash course

We are motivated to solve 

$$
\begin{cases}
\max_xf(x)\\
g(x) = 0 \\
h(x) \geq 0
\end{cases}
$$

and this can be done using Lagrange multipliers

Since $g(x)=0$ we can’t feasibly move along the feasible region from the stationary points without decreasing $f$. The stationary points are defined as $\Delta f = -\lambda \Delta g$ for some constant $\lambda$ which is known as the Lagrange multiplier.

We’ll define the Lagrange primal function

$$
L_p(x, \lambda) = f(x) + \lambda g(x)
$$

$f(x)$ is our cost function, $\lambda$  is the Lagrange multiplier and $g(x)$ is the constraint.

To find the solution $(x^*, \lambda^*)$ we’ll derive the primal function with respect to each of the variables

$$
\begin{cases}
\frac{\partial L_p}{\partial x} = 0 \\
\frac{\partial L_p}{\partial \lambda} = 0
\end{cases} \quad\text{i.e. } \nabla L_p = 0
$$

A local maximum to the constrained optimisation problem 

$$
\begin{cases}
\max_x f(x) \\
g(x) = 0
\end{cases}
$$

with Lagrange primal function

$$
L_p(x, \lambda) = f(x) + \lambda g(x)
$$

is given by $x^*$ if and only if

1. $\Delta_xL_p(x^*, \lambda^*) = 0$
2. $\Delta_\lambda L_p(x^*, \lambda^*) = 0$
3. Hessian is negative semi-definite

In practice you would solve $\Delta L_p=0$ and verify that the solution is a valid maximum.

## Example

![Untitled](Week%207%20acec4d8daa834447bd9d2bb36af00d39/Untitled%202.png)

## Inequalities in constraints, $g(x) \geq 0$

The constraint can also be an inequality meaning that the optimum is either within the feasible region $g(x) > 0$ or along the edge $g(x) = 0$.

The optimisation problem is thus

$$
\begin{cases}
\max_xf(x) \\
g(x) \geq 0
\end{cases}
$$

with Lagrange primal function

$$
L_p(x, \lambda) = f(x) + \lambda g(x)
$$

is given by $x^*$ if and only if (Karush-Kuhn-Tucker conditions) holds

1. $\Delta _xL_p(x^*, \lambda^*) = 0$
2. $\lambda^*\geq 0$
3. $\lambda^*g(x^*)=0$
4. $g(x^*) \geq0$
5. Negative definite constraints on Hessian

# Multiple constraints

If we have multiple constraints, which we had when finding the optimal separating hyperplane because we optimise for each observation, 

$$
\begin{cases}
\max_xf(x)\\
g_j(x) = 0, \quad \forall j \\
h_k(x) \geq 0, \quad \forall k 
\end{cases}
$$

The Lagrange primal function then becomes,

$$
L_p(x, \lambda, \mu) = f(x) + \sum_j \lambda_j g_j(x) + \sum_k \mu_k h_k(x)
$$

which is given by $x^*$ if and only if

1. $\Delta _xL_p(x^*, \lambda^* \mu^*) = 0$
2. $\mu^*_k \geq 0$
3. $\mu^*_kh_k(x^*)=0$
4. $h_k(x^*) \geq 0$
5. $g_j(x^*) = 0$ 
    1. Same as $\Delta_\lambda L_p(x^*, \lambda^*, \mu^*) = 0$
6. Negative definite constraints on Hessian

We can define the Lagrange dual problem since the Lagrange primal function is,

$$
L_p(x, \lambda, \mu) = f(x) + \lambda g(x) + \mu h(x)
$$

we have that (any $\mu$ and $\lambda \geq 0$)

$$
\max_x L_p (x, \lambda, \mu) \geq f(x^*) + \lambda g(x^*) + \mu h(x^*) \geq f(x^*)
$$

Define the Lagrange dual function

$$
L_D(\lambda, \mu) = \max_x L_p(x, \lambda, \mu)
$$

Optimum for $f$ if equality in

$$
\min_{\lambda \geq 0} L_D(\lambda, \mu) = \min_{\lambda \geq 0} \max_x L_P(x, \lambda) \geq f(x^*)
$$

# Using Lagrange multipliers for Optimal Separating Hyperplane

**Stage 1** Incorporate constraints using Lagrange multipliers

$$
\begin{cases}
L(\beta, \beta_0, \alpha) = \frac12 ||\beta||^2 - \sum_{i=1}^n \alpha_i(y_i(x_\beta + \beta_0) - 1)\\
\alpha_i \geq 0, \quad \forall i
\end{cases}
$$

We have one Lagrange multiplier per observation, so the sum is a sum over observations.

**Stage 2** Differentiate and set to zero

$$
\begin{cases}
\frac{\partial L}{\partial \beta} = \beta - \sum_i \alpha_i y_i x_i^T = 0\\
\frac{\partial L}{\partial\beta_0} = \sum\alpha_iy_i = 0
\end{cases}
$$

and we have

$$
\begin{cases}
\beta = \sum_i\alpha_iy_ix_i^T = 0\\
\sum\alpha_iy_i=0
\end{cases}
$$

# Overlapping labels

As data is rarely completely separable we can’t directly find the optimal separating hyperplane. For this we can find the OSH on a transformed feature space. Transformed features gives non-linear decision boundaries. With the kernel trick we can use an infinite dimensional feature expansion.

For non-linear OSH we can try basis expansion

$$
\begin{cases}
\arg\max_\alpha \alpha\mathbf{1} - \frac12 \alpha^TYXX^TY\alpha \text{ s.t. }\\
\alpha_i\geq0 \quad \forall i \\
\sum\alpha_i y_i = 0
\end{cases}
$$

which would turn into

$$
\begin{cases}
\arg\max_\alpha \alpha\mathbf{1} - \frac12 \alpha^T Yh(X)h(X)^TY\alpha  \text{ s.t. } \\
\alpha_i\geq 0 \quad \forall i \\
\sum\alpha_iy_i = 0
\end{cases}
$$

- $h(X) : R^0 \rightarrow R^M, \text{ e.g. } [x_1, x_2] \rightarrow [x_1, x^2_2, x_1x_2]$
- $h(X)h(X)^T$ is of size $n\times n$

We don’t have to explicitly write out what $h(X)$  is so we can use a number of different kernels.

Some common kernels are

- **Polynomials $K_{i,k} = (1 + x_ix_j^T)^d$**
- **Radial** $K_{i,j} = \exp(-\frac1c || x_i - x_j||^2)$
- **Gaussian $K_{i,j} = \exp(-\frac{1}{2\sigma^2} ||x_i - x_j||^2)$**
- **Neural network $K_{i,j} = \tanh(c_1x_ix_j^T + c_2)$**

Line says that the radial basis function is a good one to try out first.

# Support Vector Machine

Most classification problems have overlapping classes. If we are to try and fit the OSH we will likely overfit, so we’ll modify is to allow for some overlap in order to generalise better. This is the idea behind Support Vector Machines. Together with the kernel trick the SVM becomes a very flexible classifier.

From OSH we defined

$$
\begin{cases}
\arg\min_{\beta, \beta_0} \frac12||\beta||^2 \text{ s.t. } \\
y_i(x_i\beta-\beta_0) \geq1 \quad \forall i
\end{cases}
$$

and now for allowing some overlap

$$
\begin{cases}
\arg\min_{\beta, \beta_0} \frac12 ||\beta||^2 + \lambda \sum_{i=1}^n \xi_i \text{ s.t. }\\
y_i(x_i\beta - \beta_0) \geq 1 - \xi_i\quad\forall i \\
\xi_i\geq 0 \quad\forall i
\end{cases}
$$

We give ourselves a budget for overlap. Smaller budget, smaller $\lambda$, noisier solution.

Now solve similarly to OSH with by finding the Lagrange multipliers. The only difference is that before the Lagrange multipliers were only constrained to be $\alpha_i \geq 0 \quad \forall i$ and now they have an additional constrain $0 \leq \alpha_i \leq \lambda \quad \forall i$. Both are quadratic problems with linear constraints and only includes $X$ through inner products meaning that we can apply the kernel trick.

A popular choice for kernel is the radial basis function where we have one hyperparameter $c$ which should be searched for extensively using cross-validation due to the risk of overfitting.