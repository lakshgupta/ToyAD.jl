# addition
function ad_add(x::AD, y::AD)
  gradSize = (size(x.value) > size(y.value))? size(x.value):size(y.value)
  result = AD("+", x.value + y.value, true, zeros(gradSize), ad_addD, (x.level > y.level ? x.level:y.level)+1)
  push!(result.parents, x)
  push!(result.parents, y)
  return result
end
function ad_addD(prevGrad::Matrix{Float64}, adNodes::Array{AD, 1})
  adNodes[1].grad += 1 * prevGrad
  adNodes[2].grad += 1 * prevGrad
  return
end
+(x::AD, y::AD) = ad_add(x, y)

# multiplication
function ad_mul(x::AD, y::AD)
  if size(x.value) == (1,1) || size(y.value) == (1,1)
    return ad_elmul(x, y)
  else
    resultValue = x.value * y.value
    result = AD("*", resultValue, true, zeros(size(resultValue)), ad_mulD, (x.level > y.level ? x.level:y.level)+1)
    push!(result.parents, x)
    push!(result.parents, y)
    return result
  end
end
function ad_mulD(prevGrad::Matrix{Float64}, adNodes::Array{AD,1})
  rowNode1, colNode1 = size(adNodes[1].value)
  rowNode2, colNode2 = size(adNodes[2].value)

  for i = 1:rowNode1, j = 1:colNode2
    pd = prevGrad[i,j]
    for k = 1:colNode1
      adNodes[1].grad[i,k] += adNodes[2].value[k,j] * pd
      adNodes[2].grad[k,j] += adNodes[1].value[i,k] * pd
    end
  end
end
*(x::AD, y::AD) = ad_mul(x, y)

# element-wise multiplication
function ad_elmul(x::AD, y::AD)
  gradSize = (size(x.value) > size(y.value))? size(x.value):size(y.value)
  result = AD(".*", x.value .* y.value, true, zeros(gradSize), ad_elmulD, (x.level > y.level ? x.level:y.level)+1)
  push!(result.parents, x)
  push!(result.parents, y)
  return result
end
function ad_elmulD(prevGrad::Matrix{Float64}, adNodes::Array{AD,1})
  adNodes[1].grad += adNodes[2].value .* prevGrad
  adNodes[2].grad += adNodes[1].value .* prevGrad
  return
end
.*(x::AD, y::AD) = ad_elmul(x, y)

# relu
function reluMat(z::Matrix{Float64})
    return max(0.0, z);
end
function relu(x::AD)
  result = AD("relu", reluMat(x.value), true, zeros(size(x.value)), ad_reluD, x.level + 1)
  push!(result.parents, x)
  return result
end
function reluGradientMat(z::Matrix{Float64})
    grad = ones(z);
    grad[z.<=0] = 0;
    return grad;
end
function ad_reluD(prevGrad::Matrix{Float64}, adNodes::Array{AD,1})
  adNodes[1].grad += reluGradientMat(adNodes[1].value) .* prevGrad
  return
end

# sigmoid
function sigmoidMat(z::Matrix{Float64})
    g = 1.0 ./ (1.0 + exp(-z));
    return g;
end
function sigmoid(x::AD)
  result = AD("sigmoid", sigmoidMat(x.value), true, zeros(size(x.value)), ad_sigmoidD, x.level + 1)
  push!(result.parents, x)
  return result
end
function sigmoidGradientMat(z::Matrix{Float64})
  return z.*(1.0 - z);
end
function ad_sigmoidD(prevGrad::Matrix{Float64}, adNodes::Array{AD,1})
  adNodes[1].grad += sigmoidGradientMat(adNodes[1].value) .* prevGrad
  return
end

# tanh
function tanh(x::AD)
  result = AD("tanh", tanh(x.value), true, zeros(size(x.value)), ad_tanhD, x.level + 1)
  push!(result.parents, x)
  return result
end
function tanhGradientMat(z::Matrix{Float64})
  return (1.0 - z^2)
end
function ad_tanhD(prevGrad::Matrix{Float64}, adNodes::Array{AD,1})
  adNodes[1].grad += tanhGradientMat(adNodes[1].value) .* prevGrad
  return
end

# softmax
function softmaxMat(z::Matrix{Float64})
    g = exp(z)
    g /=sum(g)
    return g;
end
function softmax(x::AD)
    result = AD("softmax", softmaxMat(x.value), true, zeros(size(x.value)), ad_softmaxD, x.level + 1)
    push!(result.parents, x)
    return result
end
function ad_softmaxD(prevGrad::Matrix{Float64}, adNodes::Array{AD,1})
  adNodes[1].grad += adNodes[1].value - prevGrad
  return
end
