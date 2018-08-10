# PopulationDynamics.jl
[![Build Status](https://travis-ci.org/grero/PopulationDynamics.jl.svg?branch=master)](https://travis-ci.org/grero/PopulationDynamics.jl)
[![Coverage Status](https://coveralls.io/repos/github/grero/PopulationDynamics.jl/badge.svg?branch=master)](https://coveralls.io/github/grero/PopulationDynamics.jl?branch=master)
## Usage
The basic functionality of the package is captured in the function [fit_dynamics](https://github.com/grero/PopulationDynamics.jl/blob/916f924a377d4e760c7f6e36fe9f762e7d7ded34/src/fit.jl#L37) which takes an input of type `Array{T,3} where T <: Real`, which is assumed to have dimensions time by cells by trials. One can then do,

```julia
dims = size(X,2)
q = PopulationDynamics.fit_dynamics(X)
Aq = PopulationDynamics.sfunc(q.minimizer, dim)
```

which returns an `(dims+1)×(dims+1)` transformation matrix `Aq` which captures rotation, scaling and shearing in `Aq[1:dims-1, 1:dims-1]` and translation in `Aq[1:end-1, end]`.
