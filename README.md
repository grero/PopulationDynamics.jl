# PopulationDynamics.jl
[![Build Status](https://travis-ci.org/grero/PopulationDynamics.jl.svg?branch=master)](https://travis-ci.org/grero/PopulationDynamics.jl)
[![Coverage Status](https://coveralls.io/repos/github/grero/PopulationDynamics.jl/badge.svg?branch=master)](https://coveralls.io/github/grero/PopulationDynamics.jl?branch=master)
## Usage
The basic functionality of the package is captured in the function `fit_dynamics` which takes an input of type `Array{T,3} where T <: Real`, which is assumed to have dimensions time by cells by trials. One can then do,

```julia
dims = size(X,2)
q = PopulationDynamics.fit_dynamics(X)
Aq = PopulationDynamics.sfunc(q.minimizer, dim)
```

which returns an `(dims+1)×(dims+1)` transformation matrix `Aq` which captures rotation, scaling and shearing in `Aq[1:dims-1, 1:dims-1]` and translation in `Aq[1:end-1, end]`.
