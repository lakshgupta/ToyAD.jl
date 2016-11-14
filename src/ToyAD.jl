module ToyAD
  using Blink, JSON
  import Base.+, Base.(.+), Base.*, Base.(.*), Base.show

  # Automatic Differentiation
  include("Utils.jl")
  include("AD.jl")
  include("ADPlot.jl")
  include("ADOperations.jl")
  include("Backprop.jl")

  # types
  export AD

  # plotting graph
  export plot

  # operations on AD type
  export +, .+, *, .*, relu, sigmoid, tanh, softmax

  # Backprop
  export backprop

end # module
