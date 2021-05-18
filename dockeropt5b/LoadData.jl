#--------------------
#Load OptDigits Data
#--------------------

#Load entire dataset
x_total = deserialize("optdigits_x.jld")
y_total = deserialize("optdigits_y.jld")

#Standardize x Data
dx = fit(ZScoreTransform, x_total, dims=2)
StatsBase.transform!(dx, x_total)

#Load the OptDigits Data
num_samples = 50
num_classes = 5

x_train, y_train = balanced_set(x_total,y_total,num_samples,num_classes,1);#Random seed 1 for train data
x_test, y_test = balanced_set(x_total,y_total,200,num_classes,300);#Random seed 2 for test data

#Get PCA Transform for x
x = transpose(x_train)

########################################
dims = 20 #change this line for PCA dims
########################################

x_pca = fit(PCA,x,maxoutdim=dims)
xt = MultivariateStats.transform(x_pca,x)

#Testing PCA Transform
xz = MultivariateStats.transform(x_pca,transpose(x_test));


#One-Hot Encode Y
y = y_train
yt = Flux.onehotbatch(y,[:1,:2,:3,:4,:5])#,:6,:7,:8,:9,:10]);
#Test Set
yz = y_test
yzt = Flux.onehotbatch(yz,[:1,:2,:3,:4,:5])#,:6,:7,:8,:9,:10]);
