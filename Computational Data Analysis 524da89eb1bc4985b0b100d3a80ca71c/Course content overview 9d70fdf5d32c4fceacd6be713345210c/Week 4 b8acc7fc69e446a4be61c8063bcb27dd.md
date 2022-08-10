# Week 4

Exercises: https://drive.google.com/file/d/11dPpWtJjfxgKsEJFCNLVK0i35lh1CG_R/view?usp=sharing
Literature: ESL Chapters: 4.3, 4.4, 5.1, 5.2
Slides: https://drive.google.com/file/d/1K-NEQtgS8L0prjsMy4LKz6Ps9xhk69iA/view?usp=sharing
Subjects: Linear classifiers [LDA, QDA, Logistic Regression, Splines, ROC]

# Linear Discriminant Analysis

LDA is a classification method based on probability of class belonging and creating linear decision boundaries.

The probability of being in the class $k$ given the observation $x$ can be expressed as $P(G=k\vert X = x)$.

We need a stochastic model of the data to calculate probabilities and assume that the data comes from different Gaussian distributions. That is, we characterise the classes by their mean and dispersion/covariance structure. 

For simplicity, we can assume that the covariance structure is equal between classes. This allows us to do LDA.

$G(x)$ predicts class belonging for $x$,

$$
G(x) = \text{arg}\max_k P(G=k|X=x)
$$

with the probability given by Bayes theorem

$$
P(G=k|X=x) = \frac{f_k(x)\pi_k}{\sum_l^kf_l(x)\pi_l}
$$

$$
\begin{align*}
&f_l = \text{distribution of class } l \\
&\pi_l = \text{a priori probability for class } l \text{ (estimate or best guess)} \\
&\text{Total probability, } \sum\pi_l = 1
\end{align*}
$$

Missing: Odds-Ratios (slide 9) until More than two classes (slide 13)

Use plug-in estimates for unknown parameters,

$$
\begin{align*}
&\hat{\pi}_k = N_k/N, \text{where } N_k \text{ is number of class-}k \text{ observations} \\
&\hat{\mu}_k \sum_{g_i=k} x_i/N_k \\
&\hat{\sum} = \sum^K_{k=1} \sum_{g_i=k} (x_i - \hat{\mu}_k) (x_i  \hat{\mu}_k)^T / (N-K)
\end{align*}
$$

# Regularised Discriminant Analysis

## Quadratic Discriminant Analysis

Normal LDA assumes equivalence between covariance structures. When this is not the case we can use QDA. 

In LDA when $p >> n$ the covariance matrix has rank of at most $n$ and needs to be inverted. The idea to solve this is similar to ridge regression.

We go through three ways of dealing with this:

1. Make a compromise between LDA and QDA. 

$$
\hat{\sum}_k(\alpha) = \alpha\hat{\sum}_k + (1 - \alpha) \hat{\sum}
$$

1. Shrinks the covariance towards its diagonal

$$
\hat{\sum}_k(\gamma) = \gamma \hat{\sum} + (1 - \gamma) \text{diag}\left(\hat{\sum}\right)
$$

1. Shrink the covariance towards a scalar covariance structure

$$
\hat{\sum}_k(\gamma) = \gamma \hat{\sum} + (1 - \gamma)\hat{\sigma}^2 I
$$

## Reduced Rank Discriminant Analysis

The centroids lie in an affine subspace of dimension $\leq K -1$.

We want to find the maximum between-class variance while minimising the within-class variance. We then fitted a $\beta$-vector which would give this projection, this is called canonical discriminant analysis.

## Sparse Discriminant Analysis

It can be shown that optimal scoring and LDA are equivalent. 

$$
\min_{\theta_k,\beta_k} || Y\theta_k - X\beta_k||^2_2
$$

such that $\frac1n \theta^T_kY^TY\theta_k = 1, \quad \theta^T_k Y^T T \theta_I = 0 \quad \forall I < k$

This constraint means that the scoring function $Y\theta_k$ is an orthonormal basis.

As it is a regression we can add $L_1$ and $L_2$ norm

$$
\min_{\theta_k,\beta_k} || Y\theta_k - X\beta_k||^2_2 + \lambda_2 || \beta_k||^2_2 + \lambda_1||\beta_k||_1
$$

with the same constraints. This gives sparsity in the features.

This will provide a grouping of highly correlated features due to the ridge and feature selection due to the Lasso constraint.

# Logistic regression

This method is used for both classification and regression. It also has a lot less assumptions than LDA.

The thing that made LDA linear is the fact that we assume equal covariance matrices as well as the classes having Gaussian distributions. **These assumptions are thrown away in Logistic regression.**

We’ll optimise the log-odds functions directly

$$
\log \frac{P(G=g_1|X=x)}{P(G=g_2|X=x)} = \beta_0 + x\beta
$$

So the question to answer in logistic regression is: what is a good choice of $\{\beta_0, \beta\}$?

By optimising the log-likelihood (insert equation, slide 29) we’ll solve logistic regression.

The probability for a class is given by $P(G=g_i \vert X=x) = \frac{e^{\beta_0 + x\beta}}{1 + e^{\beta_0 + x\beta}}$ where the decision boundary is when $P = 0.5$ which is when $\beta_0 + x\beta = 0$.

It is shown that the LDA and logistic regression solutions are quite similar.

We can perform logistic regression for an arbitrary number of classes.

Interpretation of the coefficients in logistic regression are exemplified as follows: If we model lung cancer (yes/no) as a function of smoking (number of cigarettes per day) and we estimate a $\beta = 0.02$ we can conclude that a unit increase in smoking (one extra cigarette) means an increase in lung cancer risk (odds) of $\exp(0.02) \approx 1.02 = 2\%$.

Because of the sigmoid in logistic regression outliers don’t have much of an impact.

# Regularised Logistic regression

If we have more variables than observations, $p > n$, that would be an issue for Logistic regression. A solution to this is elastic net regularisation of the likelihood,

$$
\begin{align*}
[\beta, \beta_0] =& \text{arg}\max_{\beta, \beta_0} \{\log L(\beta, \beta_0) - P_{\lambda, \alpha}(\beta)\} \\
=& \text{arg}\max_{\beta, \beta_0} \left\{ \sum_{i=1}^n \left[y_i(\beta_0 + \beta^Tx_i) - \log(1 + e^{1 + \beta_0 + \beta^tx_i})\right] - P_{\lambda, \alpha}(\beta) \right\}
\end{align*}
$$

with $P_{\lambda, \alpha} = \lambda \left( \frac12 (1-\alpha) \vert\vert\beta\vert\vert^2_2+\alpha\vert\vert\beta\vert\vert _1 \right)$ and use cross-validation for $\lambda$ and $\alpha$.

Previously we’ve added the elastic net regularisation, here we subtract as we are maximising the log-likelihood.

We use Logistic regression because of the nice interpretations of the log-odds. It is often used as a baseline as it is simple to use and often performs very well.

# LDA vs. Logistic regression

- Logistic regression throws away the assumptions of LDA (Classes being Gaussian distributed as well as having equal covariance structures) which makes it more robust.
- Logistic regression handles categorical variables better than LDA. This is due to the Gaussian assumption and one-hot-encoded variables doesn’t make sense to assume to be Gaussian.
- Logistic regression doesn’t weigh outliers (observations far from the boundaries) much due to the sigmoid function whereas LDA weighs every observation equally making it better when classes are perfectly separated and there is no overlap.
- Logistic regression is easy to interpret due to log-odds.
- Logistic regression is often used as a baseline and performs very well.
- Logistic regression can be combined with regularisation of parameters $(n < p)$
- Logistic regression can be generalised to multi-class problems.

# Basis expansions

The idea here is to replace the variables (columns) of the data matrix, $X$, with transformations $h(X)$ s.t.

$$
y = X\beta=\sum_{i=1}^p\beta_ix_i \rightarrow\sum_{i=1}^M\beta_i'h_i(X)
$$

We can perform either linear or non-linear transformations. Linear transformations have the advantage of being easily interpretable. There are also more advanced transformations.

# Splines

Splines are simply piece-wise polynomials.

Given some function we want to approximate we can define intervals of interest where we use a hinge-function $(x-5)_+$ which is equivalent to the function $\max(0, 5)$. 

This provides a high degree of flexibility with variance under control.

![Untitled](Week%204%20b8acc7fc69e446a4be61c8063bcb27dd/Untitled.png)