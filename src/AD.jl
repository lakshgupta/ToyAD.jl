# PS: Julia data structures are columnar
# type to operate on
type AD
  name::String
  value::Matrix{Float64}
  opNode::Bool
  grad::Matrix{Float64}
  gradOp::Function
  parents::Array{AD}
  level::Int64
  output::Matrix{Float64}
  outputOrder::Int64
  loss::Float64

  AD(val::Float64) = new(@getName(val), ones(1,1)*val, false, zeros(1,1), ad_constD, Array(AD,0), 1, zeros(1,1), -1, -1)
  AD(val::Int64) = new(@getName(val), ones(1,1)*val, false, zeros(1,1), ad_constD, Array(AD,0), 1, zeros(1,1), -1, -1)
  AD(val::Matrix{Float64}) = new(@getName(val), val, false, zeros(size(val)), ad_constD, Array(AD,0), 1, zeros(1,1), -1, -1)
  AD(name::String, val::Matrix{Float64}, opNode::Bool, grad::Matrix{Float64}, gradOp::Function, level::Int64) = new(name, val, opNode, grad, gradOp, Array(AD,0), level, zeros(1,1), -1, -1)
  AD(name::String, val::Matrix{Float64}, opNode::Bool, grad::Matrix{Float64}, gradOp::Function, level::Int64, output::Matrix{Float64}, outputOrder::Int64, loss::Float64) = new(name, val, opNode, grad, gradOp, Array(AD,0), level, output, outputOrder, loss)
end

# since we do not want to differentiate with respect to a constant
function ad_constD(this::AD)
  return zeros(1,1)
end
