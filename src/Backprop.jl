function backprop(graph::AD, setGradToOne::Bool)
  # set the derivative to 1
  if setGradToOne
    graph.grad += 1
  end
  bfs = [graph]
  while length(bfs) != 0
    current = pop!(bfs)
    current.gradOp(current)
    numParents = length(current.parents)
    for i=1:numParents
      push!(bfs, current.parents[i])
    end
  end
  return graph
end
backprop(graph::AD) = backprop(graph, true)

function resetGrad(graph::AD)
  bfs = [graph]
  while length(bfs) != 0
    current = pop!(bfs)
    current.grad = current.grad .* 0
    numParents = length(current.parents)
    for i=1:numParents
      push!(bfs, current.parents[i])
    end
  end
  return graph
end
