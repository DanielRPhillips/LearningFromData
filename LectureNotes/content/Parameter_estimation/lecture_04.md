# Lecture 4: Clean up and Parameter estimation




## The near ubiquity of Gaussians

In the last lecture we considered a Gaussian distribution and
estimated its mean and variance. We are going to see a lot of Gaussian
distributions in this course. And indeed some people implicitly always
assume Gaussian distributions. So this seems like as good a place as
any to pause and consider why Gaussian distributions are a common
choice to describe noise, as well as thinking about the circumstances
in which that choice might be a poor one.

It turns out there are several reasons why one might choose a Gaussian
to describe a probability distribution. Here are two:

### The Gaussian is to statistics what the harmonic oscillator is to mechanics

Suppose we have a probability distribution $p(x | D,I)$ that is
unimodal (has only one hump), then one way to form a ``best estimate''
for the variable $x$ is to compute the maximum of the
distribution. (To save writing we denote the pdf of interest as $p(x)$
for a while hereafter.) 
```{image} /_images/point_estimate_cartoon.png
:alt: point estimate
:class: bg-primary
:width: 250px
:align: right
```
We find this point, which we'll denote by $x_0$, using calculus:

$$
  \left.\frac{dp}{dx}\right|_{x_0} = 0
  \quad \mbox{with} \quad
    \left.\frac{d^2p}{dx^2}\right|_{x_0} < 0 \ \text{(maximum)}.
$$

To characterize the posterior $p(x)$, we look nearby. We want to know
how sharp this maximum is: is $p(x)$ sharply peaked around $x=x_0$ or
is the maximum kind-of shallow? To work this out we'll do a Taylor
expansion around $x=x_0$. 
$p(x)$ itself
varies too fast, but since $p(x)$ is positive definite we can
Taylor expand $\log p$ instead. (See the box below for a strict mathematical
reason why it's a bad idea to directly Taylor expand $p(x)$ around its
maximum.)


$$
 \Longrightarrow\ L(x) \equiv \log p(x|D,I) = 
   L(x_0) + \left.\frac{dL}{dx}\right|_{x_0 = 0}
   + \frac{1}{2} \left.\frac{d^2L}{dx^2}\right|_{x_0 = 0}(x-x_0)^2 + \cdots
$$

Note that $\left.\frac{d^2L}{dx^2}\right|_{x_0 = 0} < 0$.
If we can neglect higher-order terms, then

$$
  p(x| D,I) \approx A\, e^{\frac{1}{2}\left.\frac{d^2L}{dx^2}\right|_{x_0 = 0}(x-x_0)^2} ,
$$

with $A$ a normalization factor. So in this general circumstance we get a Gaussian. Comparing to

$$
  p(x|D,I) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/\sigma^2}
  \quad\Longrightarrow\quad
  \mu = x_0, \ \sigma = \left(-\left.\frac{d^2L}{dx^2}\right|_{x_0}\right)^{-1/2}
$$

* We usually quote $x = x_0 \pm \sigma$, because *if* it is a Gaussian this is *sufficient* to tell us the entire distribution and $n$ standard deviations is $n\times \sigma$.

* But for a Bayesian, the full posterior $p(x|D,I)$ for all $x$ is
  the general result, and $x = x_0 \pm \sigma$ may be only an
  approximate characterization.

:::{admonition} To think about ...
What if $p(x|D,I)$ is asymmetric? What if it is multimodal?
:::


:::{admonition} $p$ or $\log p$?
* We motivated Gaussian approximations from a Taylor expansion to quadratic order of the *logarithm* of a pdf. 
What would go wrong if we directly expanded the pdf? Well, if we do
that we get:

$$
  p(x) \approx p(x_0) + \frac{1}{2}-\left.\frac{d^2p}{dx^2}\right|_{x_0}(x-x_0)^2
 \ \overset{x\pm\rightarrow\infty}{\longrightarrow} -\infty,
$$

i.e., we get something that diverges as $x$ tends to either plus or
minus infinity. 
```{image} /_images/pdf_expansion_cartoon.png
:alt: pdf expansion
:class: bg-primary
:width: 400px
:align: center
```

* A pdf must be normalizable and positive definite, so this approximation violates these conditions!
:::

### The Central Limit Theorem

Another reason a Gaussian pdf emerges in many calculations is because
the Central Limit Theorem states that all (or almost all) probability
distributions will eventually produce Gaussians if you take enough
data (or, equivalently, draw enough samples) from them.

_Central Limit Theorem_: The sum of $n$ random values drawn from any
pdf of finite variance $\sigma^2$ tends as $n \rightarrow \infty$ to
be Gaussian distributed about the expectation value of the sum, with
variance $n \sigma^2$.

#### Consequences:

* The mean of a large number of values becomes normally distributed
  _regardless_ of the probability distiburtion the values are drawn
  from.

* The binomial, Poisson, Student's t-, and ... distribtuions all kind
  of look liek Gaussian distributions in the limit of a large number
  of degrees of freedom.

#### Proof:

See worksheet.

#### Notebook:

Look at [](/notebooks/Basics/visualization_of_CLT.ipynb).

Things to think about:

* What does ``large'' number of degrees of freedom acutally mean? Does
it matter where we look?

* Can you identify a case where the CLT will fail?



## p-values: when all you can do is falsify

* A common way for a frequentist to discuss a theory/model, or put a
bound on a parameter value, is to quote a p-value.

* This is set up using something called the null hypothesis. Somewhat
  perversely you should pick the null hypothesis to be the opposite of
  what you want to prove. So if you want to discover the Higgs boson,
  the null hypothesis is that the Higgs boson does not exist.

* Then you pick a level of proof you are comfortable with. For the
  Higgs boson (and for many other particle physics experiments) it is
  `` 5 sigma''. How do you think we convert this statement to a
  probability?

* One minus the resulting probability is called the $p$-value. We will
  denote it $p_{\rm crit}$. There is nothing
  God-given about it. It is a standard (like ``beyond a resaonable
  doubt") that has been established in a research community for
  determining that something is (likely) going on. 

* You then take data and compute $p(D|null hypothesis)$. If $p(D|null
  hypothesis) < p_{\rm crit}$ then you conclude that the ``the null
  hypothesis is rejected at the $1- p_{\rm crit}$ level''.

* Note that if $p(D|null hypothesis) > p_{\rm crit}$ you cannot
  conculde that the null hypothesis is true. It just means ``no effect
  was observeed".

* Look at
  [](/notebooks/Basics/Bayesian_updating_coinflip_interactive.ipynb). Pick
  a $p$-value. If  $H=0.4$ work out how many coin tosses it would take
  to reject the null hypothesis that it's a fair coin ($H=0.5$) at
  that $p$-value.


## Bayesian degree of belief intervals and frequentist confidence intervals

In class on Wednesday we also talked about the difference between the
68% degree of belief interval for the most likely value (in that case
the bias weighting of the coin) and a frequentist $1 \sigma$ confidence
interval.

* First point is that $1 \sigma=68$% assumes a Gaussian distribution
  around the maximum of the posterior (cf. above). While this will
  often work out okay, it may not. And, as we seek to translate, 
  $n \sigma$ intervals into DoB statements, assuming a Gaussian
  becomes more and more questionable the higher $n$ is. (Why?)

* But the second point is more philosophical (meta-statistical?). One
  interval is a statement about $p(x|D,I)$, while the other is a
  statement about $p(D|x,I)$.

* (Note that because the conversion between the two pdfs requires the
 use of Bayes' theorem the Bayesian interval may be affected by the
 choice of the prior.)

* Bayesian version is easy; a 68% credible interval or Bayesian
  confidence interval or degree-of-belief (DoB) interval is: given
  some data and some information $I$, there is a 68% chance (probability) that the interval contains the true parameter. 

* Frequentist 68% confidence interval
    * Assuming the model (contained in $I$) and the value of the
      parameter $x$ then if we do the experiment a large number of
      times then 68% of them will produce data in that interval.
   * So the *parameter* is fixed (no pdf) and the confidence
     interval is a statement about data
   * Frequentists will try to make statements about parameters, but
     they end up a bit tangled, e.g., "There is a 68% probability
     that when I compute a confidence interval from data of this sort
     that the true value of $\theta$ will fall within the
     (hypothetical) space of observations."


* For a one-dimensional posterior that is symmetric, it is clear how to define the $d\%$ confidence interval. 
    * Algorithm: start from the center, step outward on both sides, stop when $d\%$ is enclosed.
    * For a two-dimensional posterior, need a way to integrate from the top. (Could lower a plane, as desribed below for HPD.)

* What if asymmetic or multimodal? Two of the possible choices:
    * Equal-tailed interval (central interval): the area above and below the interval are equal.
    * Highest posterior density (HPD) region: posterior density for
      every point is higher than the posterior density for any point
      outside the
      interval. [E.g., lower a horizontal line over the distribution until the desired interval percentage is covered by regions above the line.]




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

