current_dir = pwd()
app_dir = "/app"
#app_dir = "/home/jberez/Projects/MNIST/dockeropt/app"
cd(app_dir)

push!(LOAD_PATH, app_dir)
push!(LOAD_PATH, current_dir)

println("Here we go...")
flush(stdout)

using Gen
using Distributions
using LinearAlgebra
using Random
using Distances
using Flux
using StatsBase
using MultivariateStats
using Serialization
using JLD
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

#Data Hyperparameters
num_samples = 50
num_classes = 5

#NUTS hyperparameters
Δ_max = 10
acc_prob = 0.65
m = 3
m2 = 3

#Network hyperparameters
network = "classifier"

#Data hyperparameters
n = num_samples #Number of samples
c = num_classes #Number of classes
d = dims #Input dimension
N = n*c #Total samples

#Node hyperparameters
#################################################
k_range = 32 #Maximum number of neurons per layer
#################################################

k_list = [Int(i) for i in 1:k_range]

obs_master = choicemap()::ChoiceMap
for i=1:length(y)
    obs_master[(:y,i)] = y[i]
end
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
    ###################
    obs[(:k,1)] = (2*i)
    ###################
    
    #(new_start,) = generate(interpolator, (x_train,), obs)
    new_start = find_best_trace(xt,y,1000,obs)
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
    for i=1:CHAINS
        write_output(i)
    end
end
