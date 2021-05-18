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

function find_best_trace(x,y,iters,obs)
    scores = []
    single_traces = []
    for i = 1:1000
        (trace,) = generate(classifier, (x,), obs)
        push!(scores,get_score(trace))
        push!(single_traces,trace)
    end
    best_trace = single_traces[findmax(scores)[2]];
    return best_trace
end

function write_output(chain)
    filename_pre = "OPTtrace5c"
    filename_end = "output.jld"
    current_file = join([filename_pre,"$chain", filename_end])
    serialize(current_file, traces[chain])
end;

function write_acceptance()
    a_filename = "OPTAcceptanceA5c.jld"
    w_filename = "OPTAcceptanceW5c.jld"
    serialize(a_filename, a_acc)
    serialize(w_filename, w_acc)
end;

function write_data(train,test)
    train_filename = "OPT_Train5c.jld"
    test_filename = "OPT_Test5c.jld"
    serialize(train_filename,train)
    serialize(test_filename,test)
end;

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
function real_data_classifier(N::Int, modes::Int, bound::Float64, σ::Float64)
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
    colors = ["blue","green"]
    for i=1:2
        mask = [classes[j] == i for j in 1:length(classes)]
        scatter(data[:,1][mask],data[:,2][mask],c=colors[i],alpha=alpha,marker=markers[i],zorder=3)
    end
end;

function data_labeller(y::Array{Float64})
    labels = [y[i] > 0.5 ? 2 : 1 for i=1:length(y)]
    return labels
end

#New Softmax
function softmax_(arr::AbstractArray)
    ex = mapslices(x -> exp.(0.5*x),arr,dims=1) #0.5 for OptDigits c, 0.1 for b
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

#OptDigits Stuff
function balanced_set(x,y,n,c,seed=0)
    
    if seed != 0
        Random.seed!(seed)
    end
    
    shuffled_indices = shuffle(1:length(y))
    x = x[shuffled_indices,:]
    y = y[shuffled_indices]
    
    x_ordered = zeros(Float64,n*c,64)
    y_ordered = zeros(Int,n*c)
    for k=1:c
        labels = [i for i in 1:length(y) if y[i]==k]
        x_ordered[k*n-(n-1):k*n,:] = x[labels,:][1:n,:]
        y_ordered[k*n-(n-1):k*n] = y[labels][1:n]
    end
    return x_ordered, y_ordered
end

function reshape_x(x)
    n = size(x)[1]
    println(n)
    x_reshaped = zeros(Float64, 8, 8, n)
    for i=1:n
        test = reshape(x[i,:], (1,8,8))
        x_reshaped[:,:,i] = reshape(x[i,:], (8,8))
    end
    return x_reshaped
end;
    
