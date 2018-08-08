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

function compute_pca(Y::Array{T,3}, rtime::Vector{Float64}, bidx::AbstractVector{Int64};maxoutdim=3) where T <: Real
    nbins, ncells, ntrials = size(Y)
    sidx = sortperm(rtime)
    X = zeros(nbins, ncells, length(bidx))
    qm = sqrt.(maximum(Y,(1,3))[1,:,1:1])
    for i in 1:size(X,1)
       for j in 1:length(bidx)-1
            x = mean(Y[i,:,sidx[bidx[j]:bidx[j+1]-1]],2)
            X[i,:,j] = x./qm
        end
    end
    pca = fit(PCA, reshape(X,size(X,1)*size(X,3),size(X,2))';maxoutdim=maxoutdim)
end

function get_scatter(X::Array{T,3}, trial_labels::AbstractVector{Int64}) where T <: Real
    nbins, ntrials, ncells = size(X)
    labels = unique(trial_labels)
    sort!(labels)
    nlabels = length(labels)
    μ = mean(X, (1,2))[:]
    μt = zeros(nlabels,ncells)
    for (ii,l) in enumerate(labels)
        tidx = find(trial_labels.==l)
        μt[ii,:] = mean(X[:,tidx,:], (1,2))
    end
    @show size(μt) size(μ)
    d = μt .- μ[:,1:1]'
    S = zeros(ncells,ncells)
    for i in 1:nlabels
        d = μt[i,:] - μ
        S .+= d*d'
    end
    S ./= nlabels
    S
end


function get_projections(X::Array{T,3}, bins::AbstractVector{Float64}, trial_labels::AbstractVector{Int64}, rtime::Vector{Float64}) where T <: Real
    nbins,ntrials,ncells = size(X)
    labels = unique(trial_labels)
    nlabels = length(labels)
    sort!(labels)
    Xt = zeros(nlabels, ncells)
    n = zeros(nlabels, 1)
    qm = sqrt.(maximum(X,(1,2))[1,1,:])
    for (ii,l) in enumerate(labels)
        tidx = find(trial_labels.==l)
        for jj in tidx
            pp = findfirst(rtime[jj] .< bins)
            Xt[ii,:] .+= (X[pp,jj,:]./qm)
            n[ii] += 1
        end
    end
    Xt ./= n
    #find the vector the maximally spans this set, i,e, the longest vector connecting any two points
    D = Distances.pairwise(Euclidean(), Xt')
    ii = indmax(D)
    r,c = ind2sub(size(D), ii)
    V = Xt[r,:] - Xt[c,:]
    V ./= sqrt(sum(abs2, V))
    V = permutedims(V[:, 1:1, 1:1], (3,2,1))
    #find two axes that span the rest of the space
    Xq = X - sum(X.*V, 3).*V
    pca = fit(PCA, reshape(Xq, size(Xq,1)*size(Xq,2), size(Xq,3))',maxoutdim=2)
    CustomProjection(cat(2, V[:], pca.proj))
end

function func!(Xn::Array{T,3}, A::Matrix{T},Xp::Array{T,3}) where T <: Real 
    for j in 1:size(Xp,3) 
        for i in 1:size(Xp,2)
            Xn[:,i,j] = Xp[:,i,j]*A
        end
    end
end

function cfunc(A::Matrix{T}, Xp::Array{T2,3}, Xn::Array{T2,3}) where T <: Real where T2 <: Real
    nbins, ndims, ntrials = size(Xp)
    L = 0.0
    xn = zeros(T, nbins,ndims)
    for j in 1:ntrials
        A_mul_Bt!(xn, Xp[:,:,j],A)
        L += vecnorm(xn - Xn[:,:,j])
    end
    L
end

"""
Take in a vector and produce the matrix
"""
function cfunc(A::Vector{T}, Xp, Xn) where T <: Real
    ndims = size(Xp,2)-1
    nd2 = ndims*ndims
    Aa = zeros(T, ndims+1, ndims+1)
    Aa[1:end-1,1:end-1] = reshape(A[1:nd2], ndims, ndims)
    Aa[1:end-1, end] = A[nd2+1:end]
    Aa[end,end] = one(T)
    cfunc(Aa, Xp, Xn)
end

function sfunc(A::Vector{T}, ndims) where T <: Real
    nd2 = ndims*ndims
    Aa = zeros(T, ndims+1, ndims+1)
    Aa[1:end-1,1:end-1] = reshape(A[1:nd2], ndims, ndims)
    Aa[1:end-1, end] = A[nd2+1:end]
    Aa[end,end] = one(T)
    Aa
end

function test_case(σ = 0.01;RNG=MersenneTwister(rand(UInt32)))
    AA = 0.01*randn(RNG, 5,5)
    Σ = AA - AA' #skew symmetric

    Ar = expm(Σ)
    A = zeros(6,6)
    A[1:5, 1:5] = Ar
    A[end,end] = 1.0
    A[1:end-1,end] = 0.01*randn(RNG, 5)
    X = zeros(130,6,56)
    X[1,:,:,:] = 0.01*randn(RNG, 6,56)
    X[:,end,:] = 1.0
    for j in 1:size(X,3)
        for i in 2:size(X,1)
            X[i,:,j] = A*X[i-1,:,j]
            X[i,1:end-1,j] += σ*randn(RNG, 5)
        end
    end
    X, A
end

function test_cfunc1()
    X,A = test_case()
    @assert cfunc(A, X[1:end-1,:,:], X[2:end,:,:]) ≈ 3.172369176601053e-12
    Af = A[1:end-1,1:end-1][:]
    Af = [Af;A[1:end-1,end]]
    @assert cfunc(Af, X[1:end-1,:,:], X[2:end,:,:]) ≈ 3.172369176601053e-12
end


function fit_dynamics(X::Array{Float64,3}, rtime, bidx;maxoutdim=10,RNG=MersenneTwister(rand(UInt32)),show_trace=true)
    nbins, ndims, ntrials = size(X)
    if maxoutdim < ndims
        pca = compute_pca(X./sqrt.(maximum(X,(1,3))), rtime, bidx;maxoutdim=maxoutdim)
        Y = zeros(size(X,1), maxoutdim, ntrials)
         
        for i in 1:size(Y,1)
            for j in 1:size(Y,3)
                Y[i,:,j] = MultivariateStats.transform(pca, float(X[i,:,j]))
            end
        end
    else
        Y = X
        maxoutdim = ndims
    end
    #prepare for fitting 
    Xa = cat(2, Y, ones(nbins, 1, ntrials))
    func_(A) = cfunc(A,Xa[1:end-1,:,:], Xa[2:end,:,:]) 
    A0 = rand(RNG, maxoutdim^2+maxoutdim)
    q = optimize(func_, A0, LBFGS(), Optim.Options(show_trace=show_trace), autodiff=:forward)
end

end #module
