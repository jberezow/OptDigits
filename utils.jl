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
    