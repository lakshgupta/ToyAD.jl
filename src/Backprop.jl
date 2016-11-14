function backprop(graph::AD, setGradToOne::Bool)
  # set the derivative to 1
  if setGradToOne
    graph.grad += 1
  end
  bfs = [graph]
  while length(bfs) != 0
    current = pop!(bfs)
    current.gradOp(current.grad, current.parents)
    numParents = length(current.parents)
    for i=1:numParents
      push!(bfs, current.parents[i])
    end
  end
  return graph
end
