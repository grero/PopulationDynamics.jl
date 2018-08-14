function cfunc(A::Matrix{T}, Xp::Array{T2,3}, Xn::Array{T2,3}) where T <: Real where T2 <: Real
    nbins, ndims, ntrials = size(Xp)
    L = 0.0
    xn = zeros(T, nbins,ndims)
    for j in 1:ntrials
        A_mul_Bt!(xn, Xp[:,:,j],A)
        L += sum(abs2, xn - Xn[:,:,j])
    end
    L
end

function cfunc(A::Matrix{T}, Xp::Array{T2,2}, Xn::Array{T2,2}) where T <: Real where T2 <: Real
    L = sum(abs2, A_mul_Bt(Xp, A) - Xn)
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

"""
Fit the activity in `X` as `X[t,:,:] = A*X[t-1,:,:]`
"""
function fit_dynamics(X::Array{Float64,3},
                      rtime=collect(linspace(0,1,size(X,3))),
                      bidx=1:size(X,3);
                      maxoutdim=10,
                      RNG=MersenneTwister(rand(UInt32)),
                      show_trace=true)
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

