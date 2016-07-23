---
layout: post
title: "Monte Carlo Methods"

excerpt: ""

tags: [MCMC, GSoC]
categories: [MCMC, GSoC]
comments: true
---
Monte Carlo methods is a class of methods or algorithms in which we try to approximate the numerical results using repeated random sampling.
Lets us look at couple of examples to develop some intuition about Monte Carlo methods.

The first example is about famous [Monty Hall problem](https://en.wikipedia.org/wiki/Monty_Hall_problem). For those who don't know about the Monty Hall
 problem here is the statement:

> "Suppose you're on a game show, and you're given the choice of three doors: Behind one door is a car; behind the others, goats. You pick a door, say No. 1, and
the host, who knows what's behind the doors, opens another door, say No. 3, which has a goat. He then says to you, "Do you want to pick door No. 2?" Is it to your
 advantage to switch your choice?"

There are also certain standard assumptions associated with it:

- The host must always open a door that was not picked by the contestant.

- The host must always open a door to reveal a goat and never the car.

- The host must always offer the chance to switch between the originally chosen door and the remaining closed door.


Now lets try to find out the solution of above problem using Monte Carlo Method. To find the solution using Monte Carlo Methods, we need to simulate
procedure (as mentioned in statement) and calculate probabilities based on outcomes of these experiments. Don't know about you but I'm too lazy to
try simulating this experiment manually :P, so I wrote a python script which does it on my behalf ;).

~~~ python
import numpy as np
# counts the number of times we succeeded on switching
successes_on_switch = 0
prior_probs = np.ones(3)/3
door = ['d1', 'd2', 'd3']
# since door are symmetrical we can run simulation assuming we select door 1 always (without loss of generality)
# So now host can choose only door 2 and door 3
# Running simulation for 1000000 times
for car_door in np.random.choice(door, size=1000000, p=prior_probs):
    # car is behind door 
    if car_door == 'd1':
        # we choose door 1 and car is behind door 1, so success with switching is zero
        successes_on_switch += 0.0
    elif car_door == 'd2':
        # we choose door 1 and car is behind door 2, monty can choose only door 3, so success on switching
        successes_on_switch += 1.0
    elif car_door == 'd3':
        # we choose door 1 and car is behind door 3, monty can choose only door 2, so success on switching
        successes_on_switch += 1.0
success_prob_on_switch = successes_on_switch/1000000.0
print('probability of success on switching after host has opened a door is:', success_prob_on_switch)
~~~

After I ran the script I got output : **probability of success on switching after host has opened a door is: 0.666325** .
You might get a different output (because of randomness) but it will be approximately same.
And the actual solution you get by solving conditional probabilities is $$\frac{2}{3}$$ which is approximately 0.6666 .
As evident result is quite a good approximation of actual result.

The next example is about approximating value of **π**.

The method is simple : We choose a unit area square and circle inscribed in it. The areas of these will be in ratio π/4.
Now we will generate some random points and count the number of points inside the circle and the total number of points.
Ratio of the two counts is an estimate of the ratio of the two areas, which is π/4. Multiply the result by 4 to estimate π.

Here is a python code for which does the above mentioned simulation:

~~~ python
import numpy as np
x = np.random.rand(7000000) # Taking 7000000 random points in between [0, 1), for x-coordinate
y = np.random.rand(7000000) # Taking 7000000 random points in between [0, 1), for y-coordinate
points_in_circle = (np.square(x) + np.square(y) <=1).sum() # points which lie in the circle x^2 + y^2 =1 (circle centred at origin with unit radius)
pi  = 4 * points_in_circle / 7000000.0
print(u"The approximate value of π is: ", pi)
~~~

The output I got : **The approximate value of π is:  3.14158742857** which is approximately equal to value of π, which is 3.14159 .

If you observe both of the above examples, there is a nice overlapping structure to these solutions

- First define the input type and its domain for the problem

- Generate random numbers from the defined input domain

- Apply deterministic operation over these numbers to get the required result

Though the above examples are simple to solve, Monte Carlo methods are useful to obtain numerical solution to 
problems which are too complicated to be solved analytically. The most popular class of Monte Carlo methods are Monte Carlo approximations for integration
a.k.a "Monte Carlo integration".

Suppose that we are trying to estimate the integral of function $$f$$ over some domain $$D$$.

$$ I = \int_{D}f(\vec{x})d\vec{x} $$

Though these integrals can be solved analytically, and when a closed form solution does not exist, numeric integration methods can be applied.
But numerical methods quickly become intractable with even small number of dimensions which are quite common in statistics. Monte Carlo Integration
allows us to calculate an estimation of the value of integration $$I$$.

Assume that we have a [probability density function](https://en.wikipedia.org/wiki/Probability_density_function) (PDF) $$p$$ defined over the domain $$D$$. Then we can re-write the above integration $$I$$ as:

$$ I = \int_{D}\frac{f(\vec{x})}{p(\vec{x})}p(\vec{x})d\vec{x} $$

The above integration is equal to  $$E\left[ \frac{f(\vec{x})}{p(\vec{x})} \right]$$
or [expected value](https://en.wikipedia.org/wiki/Expected_value) of $$\frac{f(\vec{x})}{p(\vec{x})}$$
with respect to random variable distributed according to $$p(\vec{x})$$. 

This equality is true for any PDF on D, as long as $$p(\vec{x}) \neq 0 $$ whenever $$f(\vec{x}) \neq 0 $$.
We know that we can estimate the value of $$E[X]$$ by generating a number of random samples according to distribution of random
variable and finding their average. As more samples are taken this value is sure to converge to expected value.

In this way we can estimate the value of $$E\left[\frac{f(\vec{x})}{p(\vec{x})}\right]$$ by generating a number of random samples according to
p, computing f/p for each sample, and finding the average of these values. This process as described is what we call Monte Carlo Integration.

One might be worried, what if $$p(\vec{x}) = 0$$, but probability of generating a sample at $$p(\vec{x}) = 0$$ is 0, 
so none of our samples will cause the problem.

We can write the above procedure into following simple steps:

If itegration is of format $$I = \int_{D}F(\vec{x})d\vec{x}$$, where $$D$$ is domain

- First find volume over the domain, i.e.

$$ V = \int_{D}d\vec{x} $$

- Choose $$p(\vec{x})$$ as a uniform distribution over $$D$$ , and draw $$N$$ samples, $$x_1, x_2, x_3, .., x_N$$

- Now we can approximate $$I$$ as:

$$ I \approx V \frac{1}{N}\sum_{i=1}^{N}f{\vec{x_i}} $$

Lets use the above method and try approximating integral of $$f(x) = e^{x^2/2}$$.

$$ I = \int_{0}^{1}e^{x^2/2}dx $$

Let us define $$p(x)$$ as unfiorm distribution between 0 and 1, i.e (0, 1) .

The volume is:

$$ V = \int_{0}^{1}dx = 1$$

We will now draw N independent samples from this distribution, find the expectation of that value which will be our Monte Carlo approximation for $$I$$.

Here is a python code:

~~~ python
import numpy as np
N = 1000000  # Number of Samples we want to draw
x = np.random.rand(N)  # Drawing N samples from p(x)
Expectation = np.sum(np.exp(x*x / 2)) / N   # Taking average of those samples
print("The Monte Carlo approximation of integration of e^{x^2/2} for limits (0, 1) is:", Expectation)
~~~
The output is:**The Monte Carlo approximation of integration of e^{x^2/2} for limits (0, 1) is: 1.19477498217**. The actual value of integration which I calculated
using [WorlframAlpha](http://www.wolframalpha.com/input/?i=%5Cint_%7B0%7D%5E%7B1%7De%5E%7Bx%5E2%2F2%7D) is 1.19496.

I got a more closer estimate, when I increased the sample size to 100 million: 1.1949555144469735 .

Let us now try approximating the expected value of a [Truncated normal distribution](https://en.wikipedia.org/wiki/Truncated_normal_distribution).
The truncated normal distribution is the probability distribution of a normally distributed random variable whose value is either bounded below or above (or both)

The probability density function for Truncated normal distribution is defined as:

$$ f(x; \mu, \sigma, a, b) = \frac{\frac{1}{\sigma}\phi(\frac{x - \mu}{\sigma})}{\Phi(\frac{x - \mu}{\sigma}) - \Phi(\frac{x - \mu}{\sigma})}$$,

where $$\phi(x) = \frac{1}{\surd2\pi}exp(x/2) $$, and \Phi(.) is the cummulative density function of standard normal distribution.

Now we can approximate the expected value of the Truncated normal distribution.

We will define $$f$$ as $$f(x; \mu=3, \sigma=1, a=2, b=7)$$

expected value is given by,

$$E[x] = \int_{p(x)}p(x)xdx $$

$$E[x] = \int_{2}^{7}f(x;3, 1, 2, 7)xdx $$

and, $$V = \int_{2}^{7}dx = 5 $$

Now we will draw $$N$$ independent samples $$x_1,\cdots,x_N$$ from uniform (2, 7)

So, we can approximate expected value $$E$$ as

$$E[x] \approx \frac{V}{N}\sum_{i=1}^{N}f(x_i; 3, 1, 2, 7)*x_i$$

Here is the python code for the above procedure:

~~~ python
import scipy.stats
import numpy as np
N = 1000000  # 1 million sample size
x = 5*np.random.rand(N) + 2  # Sampling over uniform (2, 7)
pdf_vals = scipy.stats.truncnorm.pdf(x, a=2,b=7,loc=3,scale=1)  # f(x; 3, 1, 2, 7)

monte_carlo_expectation = 5 * np.sum(pdf_vals*x)/N
actual_expectation = scipy.stats.truncnorm.mean(a=2,b=7,loc=3,scale=1)

print("The monte carlo expectation is {} and the actual expectation of Truncated normal distribution f(x; 3, 1, 2, 7) is {}".format(
monte_carlo_expectation, actual_expectation))
~~~
The output of above code sample which I got was: **The monte carlo expectation is 5.365583152790689 and the actual
expectation of Truncated normal distribution f(x; 3, 1, 2, 7) is 5.373215532554829** .

## Wrapping Up

In above examples it was easy to sample from the probability distribution directly. However, in most of the practical problems the distribution we want to sample
from are far more complex. In my upcoming posts I'll cover Markov Chain Monte Carlo, Metropolis Hasting algorithm, Hamiltonian Monte Carlo and No U Turn Sampler,
which cleverly allows us to sample from sophisticated distributions.
