function compute_pca(Y::Array{T,3}, rtime::Vector{Float64}, bidx::AbstractVector{Int64};maxoutdim=3) where T <: Real
    nbins, ncells, ntrials = size(Y)
    sidx = sortperm(rtime)
    X = zeros(nbins, ncells, length(bidx))
    qm = sqrt.(maximum(abs.(Y),(1,3))[1,:,1:1])
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

"""
Test case showing simple rotational dynamics. A five dimensional system starts from an initial, random, ortogonal state and evolves through time via a transformation matrix consisting of rotation and translation.
"""
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
    evolve!(X, A,σ;RNG=RNG)
    X, A
end

function evolve!(X::Array{T1,3}, A::Matrix{T2}, σ=0.01;RNG=MersenneTwister(rand(UInt32))) where T1 <: Real where T2 <: Real
    for j in 1:size(X,3)
        for i in 2:size(X,1)
            X[i,:,j] = A*X[i-1,:,j]
            X[i,1:end-1,j] += σ*randn(RNG, 5)
        end
    end
    nothing
end

function test_cfunc1()
    X,A = test_case()
    @assert cfunc(A, X[1:end-1,:,:], X[2:end,:,:]) ≈ 3.172369176601053e-12
    Af = A[1:end-1,1:end-1][:]
    Af = [Af;A[1:end-1,end]]
    @assert cfunc(Af, X[1:end-1,:,:], X[2:end,:,:]) ≈ 3.172369176601053e-12
end

