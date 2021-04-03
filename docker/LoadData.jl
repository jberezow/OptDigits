#------------------------
#Load Boston Housing Data
#------------------------

Random.seed!(10) #Best so far: Seed 3 (see noise variance of 3.048)
data = load("boston.jld")["boston"]
data = data[shuffle(1:end), :]

# Generating test/training sets:
nrow, ncol = size(data)
nrow_test  = div(nrow, 2)
nrow_train = nrow - nrow_test

x = data[:,1:13]
y = data[:,14]

dx = fit(ZScoreTransform, x, dims=1)
StatsBase.transform!(dx, x)
dy = fit(ZScoreTransform, y, dims=1)
StatsBase.transform!(dy, y)

x_train = transpose(x[1:nrow_test,1:13])
x_test = transpose(x[nrow_test+1:nrow,1:13])
y_train = y[1:nrow_test]
y_test = y[nrow_test+1:nrow];
