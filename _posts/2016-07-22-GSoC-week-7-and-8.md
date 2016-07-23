---
layout: post
title: "Google Summer of Code week 7 and 8"

excerpt: "MY PR for No-U-Turn-Sampler (NUTS) and NUTS with dual averaging has been merged.
Apart from that I made a slight change in API of all the continuous sampling algorithms. Earlier I made the `grad_log_pdf` argument to be optional.
Also since PR which dealt with implementing `JointGaussianDistribution` has been merged, I also made changes to accommodate it."

tags: [pgmpy, GSoC]
categories: [GSoC]
comments: true
---

MY PR for No-U-Turn-Sampler (NUTS) and NUTS with dual averaging has been merged PR [#706](https://github.com/pgmpy/pgmpy/pull/706). Apart from that I made a slight
change in API of all the continuous sampling algorithms. Earlier I made the `grad_log_pdf` argument to be optional. You can either pass in a custom implementation
or otherwise it will use the gradient function in the model object. But it was a poor design choice. Not only it was making code to look ugly, it was also making
things more complex with increased number if useless checks. Other issue was that if user has to implement as custom model it will not necessarily have the gradient
method. It is rightly said simpler is better :). The current API is like:

~~~python
>>> from pgmpy.inference.continuous import NoUTurnSamplerDA as NUTSda, GradLogPDFGaussian
>>> from pgmpy.factors import JointGaussianDistribution as JGD
>>> import numpy as np
>>> mean = np.array([1, -100])
>>> covariance = np.array([[-12, 45], [45, -10]])
>>> model = JGD(['a', 'b'], mean, covariance)
>>> sampler = NUTSda(model=model, grad_log_pdf=GradLogPDFGaussian, simulate_dynamics=LeapFrog)
>>> samples = sampler.generate_sample(initial_pos=np.array([12, -4]), num_adapt=10,
...                                   num_samples=10, stepsize=0.1)
>>> samples
<generator object NoUTurnSamplerDA.generate_sample at 0x7f4fed46a4c0>
>>> samples_array = np.array([sample for sample in samples])
>>> samples_array
array([[ 11.89963386,  -4.06572636],
       [ 10.3453755 ,  -7.5700289 ],
       [-26.56899659, -15.3920684 ],
       [-29.97143077, -12.0801625 ],
       [-29.97143077, -12.0801625 ],
       [-33.07960829,  -8.90440347],
       [-55.28263496, -17.31718524],
       [-55.28263496, -17.31718524],
       [-56.63440044, -16.03309364],
       [-63.880094  , -19.19981944]])
"""
~~~

Also since PR which dealt with implementing `JointGaussianDistribution` has been merged, I also made changes to accommodate it. As you can see in the example
I have imported `JointGaussianDistribution` from factors instead of models :P. Now pgmpy supports continuous models and inference on these models using sampling
algorithms. I don't have any plans for next week, I have began working on the content for introductory blog posts and ipython notebooks maybe by next fortnight
I might finish it.
