# Lecture 5: Parameter estimation 


## Parameter estimation overview comments

* In general terms, "parameter estimation" in physics means obtaining values for parameters (i.e., constants) that appear in a theoretical model that describes data. (Exceptions exist, of course.)
* Examples:
    * couplings in a Hamiltonian
    * coefficients of a polynomial or exponential model of data
    * parameters describing a peak in a measured spectrum, such as the peak height and width (e.g., fitting a Lorentzian line shape) and the size of the background
    * cosmological parameters such as the Hubble constant 
* Conventionally this process is known as "fitting the parameters" and the goal is to find the "best fit" and maybe error bars.
* We will make particular interpretations of these phrases from our Bayesian point of view.
* Plan: set up the problem and look at how familiar ideas like "least-squares fitting" show up from a Bayesian perspective.
* As we proceed, we'll make the case that for physics a Bayesian
approach is particular well suited.

## What can go wrong in a fit?

As a teaser, let's ask: what can go wrong in a fit? 

```{image} /_images/over_under_fitting_cartoon.png
:alt: bootstrapping
:class: bg-primary
:width: 400px
:align: center
```

Bayesian methods can identify and prevent both underfitting (model is not complex enough to describe the fit data) or overfitting (model tunes to data fluctuations or terms are underdetermined, leading to them playing off each other).  
$\Longrightarrow$ we'll see how this plays out.

## Notebook: Fitting a line

Look at [](/notebooks/Parameter_estimation/parameter_estimation_fitting_straight_line_I.ipynb).

Annotations of the notebook:
* same imports as before
* assume we create data $y_{\rm exp}$ ("exp" for "experiment") from an underlying model of the form

    $$
      y_{\rm exp}(x) = m_{\rm true} x + b_{\rm true} + \mbox{Gaussian noise}
    $$

    where

    $$
     \boldsymbol{\theta}_{\rm true} = [b_{\rm true}, m_{\rm true}]
      = [\text{intercept, slope}]_{\rm true}
    $$

* The Gaussian noise is taken to have mean $\mu=0$ and standard deviation $\sigma = dy$ independent of $x$. This is implemented as
`y += dy * rand.randn(N)` (note `randn`).
* The $x_i$ points themselves are also chosen randomly according to a uniform distribution $\Longrightarrow$ `rand.rand(N)`.
* Here we are using the `numpy` random number generators while we will mostly use those from `scipy.stats` elsewhere.

The theoretical model $y_{\rm th}$ is:

$$
   y_{\rm th} = m x + b, \quad \mbox{with}\ \theta = [b, m]
$$  

So in the sense of distributions (i.e., not an algebraic equation),

$$
  y_{\rm exp} = y_{\rm th} + \delta y_{\rm exp} + \delta y_{\rm th}
$$  

* The last term, which is the model discrepancy (or "theory error") will be critically important in many applications, but has often been neglected. More on this later!
* Here we'll take $\delta y_{\rm th}$ to be negligible, which means that

    $$
      y_i \sim \mathcal{N}(y_{\rm th}(x_i;\boldsymbol{\theta}), dy^2)
    $$

    * The notation here means that the random variable $y_i$ is drawn from a normal (i.e., Gaussian) distribution with mean $y_{\rm th}(x_i;\boldsymbol{\theta})$ (first entry) and variance $dy^2$ (second entry). 
    * For a long list of other probability distributions, see Appendix A of BDA3, which is what everyone calls Ref. {cite}`gelman2013bayesian`.

* We are assuming independence here. Is that a reasonable assumption?


## Correlated posteriors

* The pdf we worked out for $p(\mu,\sigma|D,I)$ a couple of lectures
  ago is characterized by elliptical contours of equal probability
  density whose major axes are aligned with the $\mu$ and $\sigma$
  axes. (You can also look at the version obtained by sampling in a
  section appended at the end of this lecture.) 
    * This is a signal of *independent* random variables.
    * Let's look at a case where this is *not* true and then look analytically at what we should expect with correlations.

* So return to notebook [](/notebooks/Parameter_estimation/parameter_estimation_fitting_straight_line_I.ipynb)
    * Review the *statistical model*.
    * What are we trying to find? $p(\thetavec|D,I)$, just as in the other notebooks, now with $\thetavec =[b,m]$.

Comments on the notebook:
* Note that $x_i$ is alo randomly distributed uniformly.
* Log likelihood gives fluctuating results whose size depend on the # of data points $N$ and the standard deviation of the noise $dy$.

:::{admonition} Explore!
Play with the notebook and explore how the size varies with $N$ and $dy$.
:::

* Compare priors on the slope $\Longrightarrow$ uniform in $m$ vs. uniform in angle.
* Implementation of plots comparing priors:
::::{admonition} Questions for the class
* With the first set of data with $N=20$ points, does the prior matter?
* With the second set of data with $N=3$ points, does the prior matter?
:::{admonition} Answers
:class: dropdown
No!  
Yes!
:::
::::
* Note: log(posterior) = log(likelihood) + log(prior)
    * Maximum is set to 1 for plotting
    * Exponentiate: posterior = exp(log(posterior))

:::{Admonition} What does it mean that the ellipses are slanted?
&nbsp;
:::

* On the second set of data: 
    * True answers are intercept $b = 25.0$, slope $m=0.5$.
    * Flat prior gives $b = -50 \pm 75$, $m = 1.5 \pm 1, so barely at the $1\sigma$ level.
    * Symmetric prior gives $b = 25 \pm 50$, slope = 0.5 \pm 0.75, so much better.
    * Distributions are wide (only three points!).

## Likelihoods (or posteriors) with two variables with quadratic approximation

```{image} /_images/posterior_ellipse_cartoon.png
:alt: posterior ellipse
:class: bg-primary
:width: 250px
:align: left
```
Find $X_0$, $Y_0$ (best estimate) by differentiating

$$\begin{align}
 L(X,Y) &= \log p(X,Y|\{\text{data}\}, I) \\
  \quad&\Longrightarrow\quad
  \left.\frac{dL}{dX}\right|_{X_0,Y_0} = 0, \ 
  \left.\frac{dL}{dY}\right|_{X_0,Y_0} = 0
\end{align}$$

* To check reliability, Taylor expand around $L(X_0,Y_0)$:

$$\begin{align}
 L &= L(X_0,Y_0) + \frac{1}{2}\Bigl[
   \left.\frac{\partial^2L}{\partial X^2}\right|_{X_0,Y_0}(X-X_0)^2
  + \left.\frac{\partial^2L}{\partial Y^2}\right|_{X_0,Y_0}(Y-Y_0)^2 \\
  & \qquad\qquad\qquad + 2 \left.\frac{\partial^2L}{\partial X\partial Y}\right|_{X_0,Y_0}(X-X_0)(Y-Y_0)
   \Bigr] + \ldots \\
   &\equiv L(X_0, Y_0) + \frac{1}{2}Q + \ldots
\end{align}$$

It makes sense to do this in (symmetric) matrix notation:

$$
  Q = 
  \begin{pmatrix} X-X_0 & Y-Y_0 
  \end{pmatrix}
  \begin{pmatrix} A & C \\
                  C & B 
  \end{pmatrix}
  \begin{pmatrix} X-X_0 \\
                  Y-Y_0 
  \end{pmatrix}
$$

$$
 \Longrightarrow
 A = \left.\frac{\partial^2L}{\partial X^2}\right|_{X_0,Y_0},
 \quad
 B = \left.\frac{\partial^2L}{\partial Y^2}\right|_{X_0,Y_0},
 \quad
 C = \left.\frac{\partial^2L}{\partial X\partial Y}\right|_{X_0,Y_0}
$$

* So in quadratic approximation, the contour $Q=k$ for some $k$ is an ellipse centered at $X_0, Y_0$. The orientation and eccentricity are determined by $A$, $B$, and $C$.

* The principal axes are found from the eigenvectors of the Hessian matrix $\begin{pmatrix} A & C \\ C & B  \end{pmatrix}$.

$$
\begin{pmatrix}
     A & C \\
     C & B
\end{pmatrix}
\begin{pmatrix}
 x \\ y
\end{pmatrix}
=
\lambda
\begin{pmatrix}
 x \\ y
\end{pmatrix}
\quad\Longrightarrow\quad
\lambda_1,\lambda_2 < 0 \ \mbox{so $(x_0,y_0)$ is a maximum}
$$

* What is the ellipse is skewed?

```{image} /_images/skewed_ellipse_cartoon.png
:alt: posterior ellipse
:class: bg-primary
:width: 250px
:align: left
```

Look at correlation matrix

$$
 \begin{pmatrix}
 \sigma_x^2 & \sigma^2_{xy} \\
 \sigma^2_{xy} & \sigma_y^2
 \end{pmatrix}
 = - \begin{pmatrix}
     A & C \\
     C & B
     \end{pmatrix}^{-1}
$$


## Compare Gaussian noise sampling to lighthouse calculation

* Jump to the Bayesian approach in [](/notebooks/Parameter_estimation/parameter_estimation_Gaussian_noise-2.ipynb) and then come back to contrast with the frequentist approach.
* The goal is to sample a posterior $p(\thetavec|D,I)$

    $$
         p(\mu,\sigma | D, I) \leftrightarrow p(x_0,y_0|X,I)
    $$

    where $D$ on the left are the $x$ points and $D$ on the right are the $\{x_k\}$ where flashes hit.
* What do we need? From Bayes' theorem, we need 

    $$\begin{align}
      \text{likelihood:}& \quad p(D|\mu,\sigma,I) \leftrightarrow p(D|x_0,y_0,I) \\
      \text{prior:}& \quad p(\mu,\sigma|I) \leftrightarrow p(x_0,y_0|I)
    \end{align}$$

* You are generalizing the functions for log pdfs and the plotting of posteriors that are in notebook [](/notebooks/Basics/radioactive_lighthouse_exercise_key.ipynb).
* Note in [](/notebooks/Parameter_estimation/parameter_estimation_Gaussian_noise.ipynb) the functions for log-prior and log-likelihood.
    * Here $\thetavec = [\mu,\sigma]$ is a vector of parameters; cf.  $\thetavec = [x_0,y_0]$.
* Step through the set up for `emcee`.
    * It is best to create an environment that will include `emcee` and `corner`. 
    :::{hint} Nothing in the `emcee` sampling part needs to change!
    ::: 
    * More later on what is happening, but basically we are doing 50 random walks in parallel to explore the posterior. Where the walkers end up will define our samples of $\mu,\sigma$
    $\Longrightarrow$ the histogram *is* an approximation to the (unnormalized) joint posterior.
    * Plotting is also the same, once you change labels and `mu_true`, `sigma_true` to `x0_true`, `y0_true`. (And skip the `maxlike` part.)
* Analysis:
    * Maximum likelihood here is the frequentist estimate $\longrightarrow$ this is an optimization problem.
    ::::{admonition} Question
    Are $\mu$ and $\sigma$ correlated or uncorrelated?
    :::{admonition} Answer
    :class: dropdown 
    They are *uncorrelated* because the contour ellipses in the joint posterior have their major and minor axes parallel to the $\mu$ and $\sigma$ axes. Note that the fact that they look like circles is just an accident of the ranges chosen for the axes; if you changed the $\sigma$ axis range by a factor of two, the circle would become flattened.
    :::
    ::::
    * Read off marginalized estimates for $\mu$ and $\sigma$.
    

