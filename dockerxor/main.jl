current_dir = pwd()
app_dir = "/app"
cd(app_dir)

push!(LOAD_PATH, app_dir)
push!(LOAD_PATH, current_dir)

println("Here we go...")
flush(stdout)

using Gen
using Distributions
using LinearAlgebra
using Flux
using Random
using Distances
using JLD
using Serialization
using StatsBase
using BNN

include("NUTS.jl")
include("RJNUTS.jl")
include("utils.jl")
include("proposals.jl")
include("LoadData.jl");

println("Packages Loaded")
flush(stdout)

ITERS = 1000
CHAINS = Threads.nthreads()
println("Number of threads: $CHAINS")

#---------------
#Hyperparameters
#---------------

#NUTS hyperparameters
Δ_max = 1000
acc_prob = 0.65
m=3
m2=2

#Network hyperparameters
k_real = 4 #Number of hidden nodes per layer
k_vector = [0.0 for i=1:k_real]
k_vector[k_real] = 1.0

#Layer hyperparameters
l_range = 8 #Maximum number of layers in the network
l_list = [Int(i) for i in 1:l_range]
l_real = 1;

#Hyperprior Hyperparameters
αᵧ = 1 #Regression Noise Shape
βᵧ = 1 #Regression Noise Scale/Rate
α₁ = 1 #Input Weights, Biases Shape
β₁ = 1 #Input Weights, Biases Scale/Rate
α₂ = 1 #Hidden & Output Weights Shape
β₂ = k_real; #Hidden & Output Weights Scale

obs_master = choicemap()::ChoiceMap
obs_master[:y] = y_train
obs = obs_master;

#-----------
#Parallelize
#-----------

println("Initializing Traces")
println("-------------------")

traces = [[] for i=1:CHAINS]
a_acc = [[] for i=1:CHAINS]
w_acc = [[] for i=1:CHAINS]

for i=1:CHAINS
    obs[:l] = ((i-1)%8 + 1)
    #(new_start,) = generate(interpolator, (x_train,), obs)
    new_start = find_best_trace(x_train,y_train,1000,obs)
    score = get_score(new_start)
    println("Chain $i starting score: $score")
    push!(traces[i],new_start)
end

flush(stdout)

active_trace = [traces[i][1] for i=1:CHAINS]
a_active = [[] for i=1:CHAINS]
w_active = [[] for i=1:CHAINS]

obs = obs_master

#--------------
#Run Inference
#--------------
cd(current_dir)
println("Beginning Inference")
println("-------------------")
flush(stdout)

try
    Threads.@threads for i=1:CHAINS
        @inbounds for i2=1:ITERS
            active_trace[i],_,_ = RJNUTS_parallel(traces[i][i2], i, i2)
            push!(traces[i],active_trace[i])
            push!(a_acc[i],a_active[i])
            push!(w_acc[i],w_active[i])
            flush(stdout)
            if i2%5 == 0
                write_output(i)
            end
            if i2%25 == 0
                write_acceptance()
            end
        end
    end
finally
    write_output()
end
