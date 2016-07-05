---
layout: post
title: "Google Summer of Code week 5 and 6"

excerpt: "During week 5, I started working on No U Turn Sampler (NUTS). NUTS is an extension of Hamiltonian Monte Carlo that eliminates the need
to set trajectory length. NUTS recursively builds a tree in forward and backward direction proposing set of likely candidates for new
value of position and momentum and stopping automatically when it proposed values are no longer useful (doubling back).

During week 6 apart from working on NUTS with dual-averaging I also used profiling to see scope of optimizations in my current implementation."

tags: [pgmpy, GSoC]
categories: [GSoC]
comments: true
---
Mid-terms results are out. Congratulations! to all fellow GSoCer's who successfully made it through the first half.
My PR [#702](https://github.com/pgmpy/pgmpy/pull/702) has been merged which dealt with the first half of my proposed project.
During week 5, I started working on No U Turn Sampler (NUTS). NUTS is an extension of Hamiltonian Monte Carlo that eliminates the need
to set trajectory length. NUTS recursively builds a tree in forward and backward direction proposing set of likely candidates for new
value of position and momentum and stopping automatically when it proposed values are no longer useful (doubling back). With dual-averaging
algorithm stepsize can be adapted on fly, thus making possible to _run NUTS without any hand tuning at all_ :) .

I tried implementing following algorithms from the paper[1]

- Algorithm 3: Efficient No-U-Turn Sampler

- Algorithm 6: No-U-Turn Sampler with Dual Averaging

The proposed API is similar to what we have for Hamiltonian Monte Carlo. Here is a sample example on how to use NUTS

~~~python
>>> from pgmpy.inference.continuous import NoUTurnSampler as NUTS, LeapFrog
>>> from pgmpy.models import JointGaussianDistribution as JGD
>>> import numpy as np
>>> mean = np.array([1, 2, 3])
>>> covariance = np.array([[4, 0.1, 0.2], [0.1, 1, 0.3], [0.2, 0.3, 8]])
>>> model = JGD(['x', 'y', 'z'], mean, covariance)
>>> sampler = NUTS(model=model, grad_log_pdf=None, simulate_dynamics=LeapFrog)
>>> samples = sampler.sample(initial_pos=np.array([0.1, 0.9, 0.3]), num_samples=20000,stepsize=0.4)
>>> samples
rec.array([(0.1, 0.9, 0.3),
 (-0.27303886844752756, 0.5028580705249155, 0.2895768065049909),
 (1.7139810571103862, 2.809135711303245, 5.690811523613858), ...,
 (-0.7742669710786649, 2.092867703984895, 6.139480724333439),
 (1.3916152816323692, 1.394952482021687, 3.446906546649354),
 (-0.2726336476939125, 2.6230854954595357, 2.923948403903159)], 
          dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8')])
~~~

 and NUTS with dual averaging

~~~python
>>> from pgmpy.inference.continuous import NoUTurnSamplerDA as NUTSda
>>> from pgmpy.models import JointGaussianDistribution as JGD
>>> import numpy as np
>>> mean = np.array([-1, 12, -3])
>>> covariance = np.array([[-2, 7, 2], [7, 14, 4], [2, 4, -1]])
>>> model = JGD(['x', 'v', 't'], mean, covariance)
>>> sampler = NUTSda(model=model)
>>> samples = sampler.sample(initial_pos=np.array([0, 0, 0]), num_adapt=10, num_samples=10, stepsize=0.25)
>>> samples
rec.array([(0.0, 0.0, 0.0),
 (0.06100992691638076, -0.17118088764170125, 0.14048470935160887),
 (0.06100992691638076, -0.17118088764170125, 0.14048470935160887),
 (-0.7451883138013118, 1.7975387358691155, 2.3090698721374436),
 (-0.6207457594500309, 1.4611049498441024, 2.5890867012835574),
 (0.24043604780911487, 1.8660976216530618, 3.2508715592645347),
 (0.21509819341468212, 2.157760225367607, 3.5749582768731476),
 (0.20699150582681913, 2.0605044285377305, 3.8588980251618135),
 (0.20699150582681913, 2.0605044285377305, 3.8588980251618135),
 (0.085332419611991, 1.7556171374575567, 4.49985082288814)], 
          dtype=[('x', '<f8'), ('v', '<f8'), ('t', '<f8')])
~~~

Performance wise NUTS is slower than a fine tuned HMC method for a simple model like Joint Gaussian Distribution where gradients are easy to compute
because of increased number of inner products. Also in terms of memory efficiency NUTS requires to store more values of position and momentum during recursion
(when we recursively build the tree). But for complex models and models with large data(high dimensionality) NUTS is really faster than tuned HMC method.

During week 6 apart from working on NUTS with dual-averaging I also used profiling to see scope of optimizations in my current implementation.
Profiling results weren't helpful. I'll try to think of different ways on how I can reduce number of gradient computations by re-using them.

For the next week I'll write tests for NUTS and NUTS with dual-averaging.

## References and Links
- [1] [Matthew D. Hoffman, Andrew Gelman: Journal of Machine Learning Research 15 (2014) 1351-1381; Algorithm 5: Hamiltonian Monte Carlo with dual-averaging](http://www.stat.columbia.edu/~gelman/research/published/nuts.pdf)

- [2] [Pull request for NUTS](https://github.com/pgmpy/pgmpy/pull/706)
