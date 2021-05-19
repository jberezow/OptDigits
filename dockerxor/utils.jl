function layer_unpacker(i,l,k)
    if i == 1
        input_dim = d
        output_dim = k[i]
    else
        input_dim = k[i-1]
        output_dim = k[i]
    end
    return input_dim, output_dim
end

function label_output(y)
    labels = [findmax(y[:,i])[2] for i=1:length(y[1,:])]
end

function find_best_trace(x,y,iters,plot_it=false)
    obs_master = choicemap()::ChoiceMap
    obs_master[:y] = y_train
    obs = obs_master;
    best_trace, scores, mses = likelihood_regression(x,y,iters)

    if plot_it
        PyPlot.scatter(mses, scores)
        plt.title("Comparing Classifier Accuracy to Log Likelihood")
        plt.xlabel("Classifier MSE")
        plt.ylabel("Log Likelihood")
    end

    pred_y = transpose(G(x_train,best_trace))[:,1]
    best_mse = mse_scaled(pred_y, y_train)
    variance = 1/(best_trace[:τᵧ])
    println("Best noise variance: $variance")
    println("Best MSE: $best_mse")
    println("Best Score: $(get_score(best_trace))")
    println("Best layer count: $(best_trace[:l])")
    
    return best_trace
end

function likelihood_regression(x,y,iters)
    obs = obs_master;
    scores = []
    mses = []
    ls = []
    best_traces = []
    (best_trace,) = generate(interpolator, (x,), obs)
    best_score = get_score(best_trace)
    best_pred_y = transpose(G(x, best_trace))[:,1]
    best_mse = mse_unscaled(best_pred_y, y)
    
    (trace,) = generate(interpolator, (x,), obs)
    score = get_score(trace)
    pred_y = transpose(G(x, trace))[:,1]
    mse = mse_unscaled(pred_y, y)
    
    for i=1:iters
        (trace,) = generate(interpolator, (x,), obs)
        score = get_score(trace)
        pred_y = transpose(G(x, trace))[:,1]
        mse = mse_unscaled(pred_y, y)
        push!(scores,score)
        push!(mses,mse)
        if mse < best_mse
            best_mse = mse
            best_score = score
            best_trace = trace
            best_pred_y = pred_y
        end
    end
    return(best_trace, scores, mses)
end;

#Generate XOR Data
function real_data_classifier(N::Int, modes::Int, bound::Float64, σ::Float64, seed=0)
    μ₁ = [-bound, -bound]
    μ₂ = [-bound, bound]
    μ₃ = [bound, bound]
    μ₄ = [bound, -bound]
    #μ₅ = [1.25, 1.25]
    #μ₆ = [1.25, 1.75]
    #μ₇ = [1.75, 1.75]
    #μ₈ = [1.75, 1.25]
    μ = [μ₁, μ₂, μ₃, μ₄]
    Σ = [[σ, 0] [0, σ]]
    
    all_samples = zeros(Float64, (N*modes, 2))
    classes = zeros(Int, (N*modes))
    
    if seed != 0
        Random.seed!(seed)
    end
    
    for i = 1:modes
        dist = MvNormal(μ[i], Σ)
        sample = rand(dist, N)::Matrix
        #scatter(sample[1,:],sample[2,:])
        all_samples[(i-1)*N+1:i*N,:] = transpose(sample)
        classes[(i-1)*N+1:i*N] = fill(i, N)
        classes = float(classes)
    end
    return all_samples, classes
end;

function plot_data_classifier(data,classes,alpha=1.0)
    markers = ["o","*"]
    colors = ["blue","red"]
    for i=1:2
        mask = [classes[j] == i for j in 1:length(classes)]
        scatter(data[:,1][mask],data[:,2][mask],c=colors[i],alpha=alpha,marker=markers[i],zorder=3)
    end
end;

function data_labeller(y::Array{Float64})
    labels = [y[i] > 0.5 ? 2 : 1 for i=1:length(y)]
    return labels
end

function write_output(chain)
    filename_pre = "XORtrace"
    filename_end = "output.jld"
    current_file = join([filename_pre,"$chain", filename_end])
    serialize(current_file, traces[chain])
end;

function write_acceptance()
    a_filename = "XORAcceptanceA.jld"
    w_filename = "XORAcceptanceW.jld"
    serialize(a_filename, a_acc)
    serialize(w_filename, w_acc)
end;

#New Softmax
function softmax_(arr::AbstractArray)
    ex = mapslices(x -> exp.(0.1.*x),arr,dims=1)
    rows, cols = size(arr)
    val = similar(ex)
    for i in 1:cols
        s = sum(ex[:,i])
        for j in 1:rows
            val[j,i] = ex[j,i]/s
        end
    end
    return val
end;
    
