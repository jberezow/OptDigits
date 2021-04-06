import Random

import MLDatasets
train_x, train_y = MLDatasets.MNIST.traindata()

function load_mnist_train_set()
    train_x, train_y = MLDatasets.MNIST.traindata()
    N = length(train_y)
    x = zeros(Float64, N, 784)
    y = Vector{Int}(undef, N)
    for i=1:N
        x[i, :] = reshape(train_x[:,:,i], (28*28))
        y[i] = train_y[i]+1
    end
    x, y
end

function load_mnist_test_set()
    test_x, test_y = MLDatasets.MNIST.testdata()
    N = length(test_y)
    x = zeros(Float64, N, 784)
    y = Vector{Int}(undef, N)
    for i=1:N
        x[i, :] = reshape(test_x[:,:,i], (28*28))
        y[i] = test_y[i]+1
    end
    x, y
end

function balanced_set(x,y,n,seed=0)
    
    if seed != 0
        Random.seed!(seed)
    end
    
    shuffled_indices = shuffle(1:length(y))
    #x = x[shuffled_indices,:]
    #y = y[shuffled_indices]
    
    x_ordered = zeros(Float64,n*10,784)
    y_ordered = zeros(Int,n*10)
    for k=1:10
        labels = [i for i in 1:length(y) if y[i]==k]
        x_ordered[k*n-(n-1):k*n,:] = x[labels,:][1:n,:]
        y_ordered[k*n-(n-1):k*n] = y[labels][1:n]
    end
    return x_ordered, y_ordered
end

function reshape_x(x)
    n = size(x)[1]
    x_reshaped = zeros(Float64, 28, 28, n)
    for i=1:n
        test = reshape(x[i,:], (1,28,28))
        x_reshaped[:,:,i] = reshape(x[i,:], (28,28))
    end
    return x_reshaped
end