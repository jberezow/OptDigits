#-------------------------
#Layer Birth / Death Moves
#-------------------------

function obs_maker(trace)
    obs_new = choicemap()::ChoiceMap
    obs_new[:y] = trace[:y]
    obs_new[:τᵧ] = trace[:τᵧ]

    obs_new[:l] = trace[:l]
    return obs_new
end

function layer_change(trace)
    current_l = trace[:l]
    if current_l == last(l_list)
        new_trace = layer_death(trace)
    elseif current_l == l_list[1]
        new_trace = layer_birth(trace)
    else
        new_trace = bernoulli(0.5) ? layer_birth(trace) : layer_death(trace)
    end
    return new_trace        
end

@gen function layer_birth(trace)
    
    previous_l = trace[:l]
    new_l = l_list[previous_l+1]
    difference = abs(new_l - previous_l)
    
    #Select Insertion Place for New Layer
    insert = previous_l + 1
    
    obs_new = obs_maker(trace)
    args = get_args(trace)
    argdiffs = map((_) -> NoChange(), args)
    obs_new[:l] = new_l
    
    #Recast Output Layer
    obs_new[(:k,new_l+1)] = trace[(:k,previous_l+1)]
    obs_new[(:W,new_l+1)] = trace[(:W,previous_l+1)]
    obs_new[(:b,new_l+1)] = trace[(:b,previous_l+1)]
    
    #Modify Weight Matrices and Bias Vector
    #------------------------------------------------
    q_score = 0
    
    for i=1:new_l
        if i == new_l
            k = trace[(:k,i-1)]
            obs_new[(:k,i)] = k
            h = Int(k * k)
            
            #Hidden Weights
            u = zeros(h)
            σ = 1/trace[(:τ,i)]
            S = Diagonal([σ for i=1:length(u)])
            W = @trace(mvnormal(u,S), (:W,i))

            #Hidden Biases
            ub = zeros(k)
            σᵦ = 1/trace[(:τᵦ,i)]
            Sb = Diagonal([σᵦ for i=1:length(ub)])    
            b = @trace(mvnormal(ub,Sb), (:b,i))
            
            obs_new[(:W,i)] = W
            obs_new[(:b,i)] = b
            obs_new[(:τ,i)] = trace[(:τ,i)]
            obs_new[(:τᵦ,i)] = trace[(:τᵦ,i)]
            
            q_score += (
                log(pdf(MvNormal(u,S),W)) + 
                log(pdf(MvNormal(ub,Sb),b))
                )
        else
            obs_new[(:k,i)] = trace[(:k,i)]
            obs_new[(:W,i)] = trace[(:W,i)]
            obs_new[(:b,i)] = trace[(:b,i)]
            obs_new[(:τ,i)] = trace[(:τ,i)]
            obs_new[(:τᵦ,i)] = trace[(:τᵦ,i)]
        end
    end
    #------------------------------------------------
    
    #Update Trace and Return Trace and Weights
    (new_trace,_,_,_) = update(trace, args, argdiffs, obs_new)
    q = -q_score
        
    return (new_trace, q)
    
end

@gen function layer_death(trace)
    
    previous_l = trace[:l] #4
    new_l = l_list[previous_l-1] #3 
    difference = abs(new_l - previous_l) #1
    
    #Select Insertion Place for New Layer
    output = previous_l + 1 #5
    
    obs_new = obs_maker(trace)
    args = get_args(trace)
    argdiffs = map((_) -> NoChange(), args)
    obs_new[:l] = new_l
    
    #Recast Output Layer
    obs_new[(:k,new_l+1)] = trace[(:k,output)] #Layer 4 = Layer 5
    obs_new[(:W,new_l+1)] = trace[(:W,output)]
    obs_new[(:b,new_l+1)] = trace[(:b,output)]
    
    #Capture Deleted Layer Weight Matrices and Bias Vector
    #-----------------------------------------------------
    q_score = 0
    
    for i=1:new_l
        if i == new_l
            k = trace[(:k,i+1)]
            h = Int(k * k)
            
            #Hidden Weights
            W = trace[(:W,output)]
            u = zeros(length(W))
            σ = 1/trace[(:τ,i)]
            S = Diagonal([σ for i=1:length(u)])

            #Hidden Biases
            b = trace[(:b,output)]
            ub = zeros(length(b))
            σᵦ = 1/trace[(:τᵦ,i)]
            Sb = Diagonal([σᵦ for i=1:length(b)])  
            
            q_score += (
                log(pdf(MvNormal(u,S),W)) + 
                log(pdf(MvNormal(ub,Sb),b))
                )
        else
            obs_new[(:k,i)] = trace[(:k,i)]
            obs_new[(:W,i)] = trace[(:W,i)]
            obs_new[(:b,i)] = trace[(:b,i)]
            obs_new[(:τ,i)] = trace[(:τ,i)]
            obs_new[(:τᵦ,i)] = trace[(:τᵦ,i)]
        end
    end
    #-----------------------------------------------------
    
    #Update Trace and Return Trace and Weights
    (new_trace,_,_,_) = update(trace, args, argdiffs, obs_new)
    #(new_trace,) = generate(interpolator, (x,), obs_new)
    q = q_score
        
    return (new_trace, q)
    
end
