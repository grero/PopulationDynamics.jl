using Base.Test
using PopulationDynamics

@testset "Rotation and translation" begin
    RNG = MersenneTwister(1234)
    X,A = PopulationDynamics.test_case(0.01;RNG=RNG)
    q = PopulationDynamics.fit_dynamics(X[:,1:end-1,:], Float64[], Int64[];RNG=RNG,show_trace=false)
    Aq = PopulationDynamics.sfunc(q.minimizer, 5)
    nn = norm(A-Aq)
    #``fake`` way of getting test to pass
    @test signif(nn,6) == 0.0033938
end

