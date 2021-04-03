function find_best_trace(x,y,iters,obs=obs_master)
    (best_trace,) = generate(interpolator, (x,), obs)
    best_pred_y = transpose(G(x, best_trace))[:,1]
    best_mse = mse_scaled(best_pred_y, y)
    
    (trace,) = generate(interpolator, (x,), obs)
    pred_y = transpose(G(x, trace))[:,1]
    mse = mse_scaled(pred_y, y)
    
    for i=1:iters
        (trace,) = generate(interpolator, (x,), obs)
        pred_y = transpose(G(x, trace))[:,1]
        mse = mse_scaled(pred_y, y)
        if mse < best_mse
            best_mse = mse
            best_trace = trace
        end
    end
    return best_trace
end

function mse_scaled(y_pred,y_real)
    y_pred = StatsBase.reconstruct(dy,y_pred)
    y_real = StatsBase.reconstruct(dy,y_real)
    √(sum((y_pred .- y_real).^2))/length(y_real)
end

function mse_unscaled(y_pred,y_real)
    √(sum((y_pred .- y_real).^2))/length(y_real)
end

function write_output(chain)
    filename_pre = "trace"
    filename_end = "output.jld"
    current_file = join([filename_pre,"$chain", filename_end])
    serialize(current_file, traces[chain])
end;

function write_acceptance()
    a_filename = "AcceptanceA.jld"
    w_filename = "AcceptanceW.jld"
    serialize(a_filename, a_acc)
    serialize(w_filename, w_acc)
end;

function likelihood_regression(x,y,iters,obs)
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
    return best_trace
end;