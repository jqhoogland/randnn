# Random neural networks

A library for sampling random neural networks.

### The model
```math
\tau \frac{dh_i}{dt}=-h_i+\sum_{j=1}^N J_{ij}\phi(h_j),
```
where:
- $`h_i(t)`$ is the state of neuron $i$ at time $t$.
- $`J_{ij}`$ is the weight of the connection from neuron $j$ to neuron $i$.
  - It is sampled from $\mathcal{N}(0, g^2/N)$, where $g$ is our coupling strength.
  - $`J_{ii}=0`$
- $`\phi`$ is the activation function ($`=\tanh`$)

