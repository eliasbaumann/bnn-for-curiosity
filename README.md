# bnn-for-curiosity
Bayesian neural networks for curiosity using action prediction

This project did not work out, in particular likely because:
- The dynamics network is trained using variance and will in turn minimize its variance
- Variance might (for this specific example and using code from Burda et al. 2018) not be a good reward to train an agent on

Large parts of this repository are taken from: https://github.com/openai/large-scale-curiosity
