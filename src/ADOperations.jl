# transpose
function ad_transpose(x::AD)
  x.value = (x.value)'
  x.grad = (x.grad)'
  return x
end
transpose(x::AD) = ad_transpose(x)

# addition
function ad_add(x::AD, y::AD)
  gradSize = (size(x.value) > size(y.value))? size(x.value):size(y.value)
  result = AD("+", x.value + y.value, true, zeros(gradSize), ad_addD, (x.level > y.level ? x.level:y.level)+1)
  push!(result.parents, x)
  push!(result.parents, y)
  return result
end
function ad_addD(this::AD)
  prevGradSize = size(this.grad)
  adNode1Size = size(this.parents[1].value)
  adNode2Size = size(this.parents[2].value)

  if adNode1Size == prevGradSize
    this.parents[1].grad += 1 * this.grad
  elseif adNode1Size[1] == prevGradSize[1]
    this.parents[1].grad += 1 * sum(this.grad,2)
  else
    this.parents[1].grad += 1 * sum(this.grad,1)
  end

  if adNode2Size == prevGradSize
    this.parents[2].grad += 1 * this.grad
  elseif adNode2Size[1] == prevGradSize[1]
    this.parents[2].grad += 1 * sum(this.grad,2)
  else
    this.parents[2].grad += 1 * sum(this.grad,1)
  end
  return
end
+(x::AD, y::AD) = ad_add(x, y)

function ad_eladd(x::AD, y::AD)
  gradSize = (size(x.value) > size(y.value))? size(x.value):size(y.value)
  result = AD(".+", x.value .+ y.value, true, zeros(gradSize), ad_eladdD, (x.level > y.level ? x.level:y.level)+1)
  push!(result.parents, x)
  push!(result.parents, y)
  return result
end
function ad_eladdD(this::AD)
  prevGradSize = size(this.grad)
  adNode1Size = size(this.parents[1].value)
  adNode2Size = size(this.parents[2].value)

  if adNode1Size == prevGradSize
    this.parents[1].grad += 1 * this.grad
  elseif adNode1Size[1] == prevGradSize[1]
    this.parents[1].grad += 1 * sum(this.grad,2)
  else
    this.parents[1].grad += 1 * sum(this.grad,1)
  end

  if adNode2Size == prevGradSize
    this.parents[2].grad += 1 * this.grad
  elseif adNode2Size[1] == prevGradSize[1]
    this.parents[2].grad += 1 * sum(this.grad,2)
  else
    this.parents[2].grad += 1 * sum(this.grad,1)
  end

  return
end
.+(x::AD, y::AD) = ad_eladd(x, y)

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
function ad_mulD(this::AD)
  rowNode1, colNode1 = size(this.parents[1].value)
  rowNode2, colNode2 = size(this.parents[2].value)

  for i = 1:rowNode1, j = 1:colNode2
    pd = this.grad[i,j]
    for k = 1:colNode1
      this.parents[1].grad[i,k] += this.parents[2].value[k,j] * pd
      this.parents[2].grad[k,j] += this.parents[1].value[i,k] * pd
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
function ad_elmulD(this::AD)
  this.parents[1].grad += this.parents[2].value .* this.grad
  this.parents[2].grad += this.parents[1].value .* this.grad
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
function ad_reluD(this::AD)
  this.parents[1].grad += reluGradientMat(this.parents[1].value) .* this.grad
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
function ad_sigmoidD(this::AD)
  this.parents[1].grad += sigmoidGradientMat(this.parents[1].value) .* this.grad
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
function ad_tanhD(this::AD)
  this.parents[1].grad += tanhGradientMat(this.parents[1].value) .* this.grad
  return
end

# softmax
function softmaxMat(z::Matrix{Float64}, dimOrder::Int64)
    g = exp(z)
    g ./=sum(g, dimOrder)
    return g;
end

# Using cross-entropy loss function.
# evaluating d(loss)/d(loss)*d(loss)/d(softmax)*d(softmax)/d(z)
# because calculating d(softmax)/d(z) individually requires too much memory.
function softmaxLoss(y::Matrix{Float64}, x::AD, dimOrder::Int64, computeLoss::Bool, reg::Float64)
   # dimOrder: along which the dimentions/classes are present
   probs = softmaxMat(x.value, dimOrder)
   # compute the loss: average cross-entropy loss and regularization
   loss = 0.0
   if computeLoss
     dataInstanceOrder = (dimOrder == 1? 2 : 1)
     numInstance = size(y, dataInstanceOrder)
     correctLogProbs = zeros(size(y))
     for j in 1:numInstance
       correctLogProbs[j, :] = -log(probs[j,Int64(dataInstanceOrder==1?y[j, 1]:y[1,j])]);
     end
     loss = sum(correctLogProbs)/numInstance + reg
   end
   
   # return node
   result = AD("softmaxLoss", probs, true, zeros(size(x.value)), ad_softmaxLossD, x.level + 1, y, dataInstanceOrder, loss)
   push!(result.parents, x)
   return result
end

softmaxLoss(y::Matrix{Float64}, x::AD, dimOrder::Int64, computeLoss::Bool) = softmaxLoss(y, x, dimOrder, computeLoss, 0.0)
softmaxLoss(y::Matrix{Float64}, x::AD, dimOrder::Int64) = softmaxLoss(y, x, dimOrder, false, 0.0)

function ad_softmaxLossD(this::AD)
  this.parents[1].grad += this.value .* this.grad
  if this.outputOrder == 1
    for j in 1:size(this.parents[1].grad, this.outputOrder)
      this.parents[1].grad[j,Int64(this.output[j,1])] -= 1;
    end
  elseif this.outputOrder == 2
    for j in 1:size(this.parents[1].grad, this.outputOrder)
      this.parents[1].grad[Int64(this.output[1,j]), j] -= 1;
    end
  end
  this.parents[1].grad /= size(this.output, this.outputOrder);
  return
end
