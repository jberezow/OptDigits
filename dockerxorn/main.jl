current_dir = pwd()
app_dir = "/app"
#app_dir = "/home/jberez/Projects/OptDigits/dockerxorn/app"
cd(app_dir)

push!(LOAD_PATH, app_dir)
push!(LOAD_PATH, current_dir)

println("Here we go...")
flush(stdout)

using Gen
using LinearAlgebra
using Random
using Distributions
using Distances
using Flux
using JLD
using MultivariateStats
using Serialization
using StatsBase
using BNN

include("NUTS.jl")
include("RJNUTS.jl")
include("utils.jl")
include("proposals.jl");

println("Packages Loaded")
flush(stdout)

ITERS = 1000
CHAINS = Threads.nthreads()
println("Number of threads: $CHAINS")

#---------------
#Hyperparameters
#---------------

#NUTS hyperparameters
Δ_max = 1
acc_prob = 0.65
m = 1
m2 = 1

#Select Network Goal
network = "classifier"

#Data hyperparameters
c = 2
n = 50 #Number of samples per mode (classifier)
m = 4 #Number of modes (classifier)
d = 2 #Input dimension
N = n*m #Total samples
σₐ = 0.1 #Mode variance (classifier)
bound = 0.5

#Data
x_raw, classes = real_data_classifier(Int(N/4), 4, bound, σₐ);
classes = [(i+1) % 2 + 1 for i in classes]
y = classes
xt = transpose(x_raw)

#One-Hot Encode Y
yt = Flux.onehotbatch(y,[:1,:2]);

#Test Set
x_raw_test, classes_test = real_data_classifier(Int(N/4), 4, bound, σₐ);
xt_test = transpose(x_raw_test)
classes_test = [(i+1) % 2 + 1 for i in classes]
y_test = classes_test
yz = y_test
yzt = Flux.onehotbatch(yz,[:1,:2]);

#Node hyperparameters
#################################################
k_range = 16 #Maximum number of neurons per layer
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
    obs[(:k,1)] = i
    (new_start,) = generate(classifier, (xt,), obs)
    #new_start = find_best_trace(xt,y,1000,obs)
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
