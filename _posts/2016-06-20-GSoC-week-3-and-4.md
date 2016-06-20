---
layout: post
title: "Google Summer of Code week 3 and 4"

excerpt: "In terms of progress week 3 turned to be dull. I was having doubts in my mind regarding the representation since start of 
coding period and somehow I always forgot to have my doubts cleared in meeting. 
I wasted a lot of time reading theory to help me out with this doubt of mine and till mid of the week 4 it remained unclear."

tags: [pgmpy, GSoC]
categories: [GSoC]
comments: true
---
In terms of progress week 3 turned to be dull. I was having doubts in my mind regarding the representation since start of 
coding period and somehow I always forgot to have my doubts cleared in meeting. 
I wasted a lot of time reading theory to help me out with this doubt of mine and till mid of the week 4 it remained unclear. 

During week 3, I re-structured my code in different parts. Remove the `BaseHamiltonianMC` and created a `HamiltonianMC` class which 
returned samples using Simple Hamiltonian Monte Carlo. This class was then inherited by `HamiltonianMCda` 
which returned samples using Simple Hamiltonian Monte Carlo. Wrote function for some section of overlapping code, 
changed name of some parameters to specify their context in a better manner. 
Apart from that I was experimenting a bit with the API and how samples should be returned. 

As discussed in last post, the parameterization of model was still unclear to me, 
but upon discussion with my mentor and other members I found that we already had a representation finalized 
for Continuous factor and Joint distributions. I wasted a lot of time on this matter, I laughed at my silly mistake. 
If I had my doubts clear in start I would have already finished with my work. No frets now it gave me a good learning experience. 
So I re-wrote my certain part of code to take this parameterization into account. In discussion of week 4 meeting upon my 
suggestion we decided to use `numpy.recarray` objects instead of `pandas.DataFrame` as `pandas.DataFrame` was adding a 
dependency and was also slower than `numpy.recarray` objects. I also improved the documentation of my code during the week 4, 
which earlier wasn’t consistent with my examples. I was allowing user to pass any n-dimensional array 
instead of mentioned 1d array in documentation, I thought it will provide more flexibility but actually it was making things ambiguous. 
At the end of week 4 the code looks really different from what it was in the start. I wrote `_sample` method which run a 
single iteration of sampling using Hamiltonian Monte Carlo. Now the code returns samples in two different types. 
If user has an installation of pandas, it returns `pandas.DataFrame` otherwise it returns `numpy.recarry` object. 
This is how output looks like now:

- If user doesn’t have a installation of pandas in environment

~~~python
>>> from pgmpy.inference.continuous import HamiltonianMC as HMC, LeapFrog
>>> from pgmpy.models import JointGaussianDistribution as JGD
>>> import numpy as np
>>> mean = np.array([-3, 4])
>>> covariance = np.array([[3, 0.7], [0.7, 5]])
>>> model = JGD(['x', 'y'], mean, covariance)
>>> sampler = HMC(model=model, grad_log_pdf=None, simulate_dynamics=LeapFrog)
>>> samples = sampler.sample(initial_pos=np.array([1, 1]), num_samples = 10000,
...                          trajectory_length=2, stepsize=0.4)
>>> samples
array([(5e-324, 5e-324), (-2.2348941964735225, 4.43066330647519),
       (-2.316454719617516, 7.430291195678112), ...,
       (-1.1443831048872348, 3.573135519428842),
       (-0.2325915892988598, 4.155961788010201),
       (-0.7582492446601238, 3.5416519297297056)], 
      dtype=[('x', '<f8'), ('y', '<f8')])

>>> samples = np.array([samples[var_name] for var_name in model.variables])
>>> np.cov(samples)
array([[ 3.0352818 ,  0.71379304],
       [ 0.71379304,  4.91776713]])
>>> sampler.accepted_proposals
9932.0
>>> sampler.acceptance_rate
0.9932
~~~

- If user has a pandas installation

~~~python
>>> from pgmpy.inference.continuous import HamiltonianMC as HMC, GradLogPDFGaussian, ModifiedEuler
>>> from pgmpy.models import JointGaussianDistribution as JGD
>>> import numpy as np
>>> mean = np.array([1, -1])
>>> covariance = np.array([[1, 0.2], [0.2, 1]])
>>> model = JGD(['x', 'y'], mean, covariance)
>>> sampler = HMC(model=model)
>>> samples = sampler.sample(np.array([1, 1]), num_samples = 5,
...                          trajectory_length=6, stepsize=0.25)
>>> samples
               x              y
0  4.940656e-324  4.940656e-324
1   1.592133e+00   1.152911e+00
2   1.608700e+00   1.315349e+00
3   1.608700e+00   1.315349e+00
4   6.843856e-01   6.237043e-01
~~~

In contrast to earlier output which was just a list of numpy.array objects

~~~python
>>> from pgmpy.inference.continuous import HamiltonianMC as HMC, GradLogPDFGaussian, ModifiedEuler
>>> from pgmpy.models import JointGaussianDistribution as JGD
>>> import numpy as np
>>> mean = np.array([1, -1])
>>> covariance = np.array([[1, 0.2], [0.2, 1]])
>>> model = JGD(['x', 'y'], mean, covariance)
>>> sampler = HMC(model=model, grad_log_pdf=GradLogPDFGaussian, simulate_dynamics=ModifiedEuler)
>>> samples = sampler.sample(np.array([1, 1]), num_samples = 5,
...                          trajectory_length=6, stepsize=0.25)
>>> samples
[array([[1],
        [1]]),
 array([[1],
        [1]]),
 array([[ 0.62270104],
        [ 1.04366093]]),
 array([[ 0.97897949],
        [ 1.41753311]]),
 array([[ 1.48938348],
        [ 1.32887231]])]
~~~

Next week I’ll try to do some changes mentioned by my mentor on my PR. Also I’ll write more test cases to individually
test each function instead of testing the overall implementation. After my PR gets merged I’ll try to write introductory blogs
related to Markov Chain Monte Carlo and Hamiltonian Monte Carlo and will work on No U Turn Sampling.
