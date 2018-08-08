module PopulationDynamics
using Distances
using StatsBase
using MultivariateStats
import MultivariateStats.transform
using Optim

struct CustomProjection
    proj::Matrix{Float64}
end

MultivariateStats.transform(X::CustomProjection, x::Vector{Float64}) = X.proj'*x
function MultivariateStats.transform(X::Union{PCA,CustomProjection}, x::Matrix{Float64})
    y = zeros(size(X.proj,2),size(x,2))
    for i in 1:size(x,2)
        y[:,i] = transform(X, x[:,i])
    end
    y
end

include("utils.jl")
include("fit.jl")
end #module
