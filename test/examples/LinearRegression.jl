#Pkg.add("PlotlyJS")
using Plots
plotlyjs()

# read the data
data = readdlm("test/examples/data/regression/ex1data1.txt", ',');

# try plotting some samples
scatter!(data[50,:], markersize=3, c=:blue)

# here we want to predict the line
# y = mx + c = theta1*x + theta2
function update(t)
