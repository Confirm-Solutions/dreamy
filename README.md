# Fluent dreaming for language models.

The code here implements the algorithms in the paper ["Fluent dreaming for language models"](https://arxiv.org/abs/2402.01702).

[Please also see the companion page that demonstrates using the code here.](https://confirmlabs.org/posts/dreamy.html)

[There is also a Colab version of the companion page.](https://colab.research.google.com/drive/1B0dM7du91BUkT7tSICXjKL7lrBAEdSa-?usp=sharing)

Modules:
- `dreamy.epo`: The main EPO algorithm along with a few utilities for loading models and constructing Pareto frontiers.
- `dreamy.attribution`: Code for creating causal attribution figures.
- `dreamy.runners`: "Runners" for different optimization targets like neurons, output logits, etc.
- `dreamy.experiment`: Code we used in writing the paper for running experiments on Modal and using S3 for storage.
