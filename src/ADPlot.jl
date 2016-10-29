type ADMetadata
  valueSize::Tuple{Int64, Int64}
  opNode::Bool
  gradSize::Tuple{Int64, Int64}
end
type ADNodesMetadata
  id::String
  name::String
  metadata::ADMetadata
  group::Int64

  ADNodesMetadata(node::AD) = new(string(pointer(node.value)), node.name, ADMetadata(size(node.value), node.opNode, size(node.grad)), node.opNode?1:2)
end
type ADLinksMetadata
  source::String
  target::String
  value::Int64
end
type D3Data
  nodes::Array{ADNodesMetadata}
  links::Array{ADLinksMetadata}
end

# traverse the graph top-down
# convert AD to ADMetadata
function getGraphMetadata(graph::AD)
  nodes = Array(ADNodesMetadata,0)
  links = Array(ADLinksMetadata,0)

  bfs = [graph]
  while length(bfs) != 0
    current = pop!(bfs)
    push!(nodes, ADNodesMetadata(current))
    currentPtr = string(pointer(current.value))
    numParents = length(current.parents)
    for i=1:numParents
      push!(links, ADLinksMetadata(currentPtr, string(pointer(current.parents[i].value)), current.level))
      push!(bfs, current.parents[i])
    end
  end

  return D3Data(nodes, links)
end

function d3ForcedDirectedGraphHTML(data::D3Data, supportsHTML::Bool)
  d3HTML =
    """
      <!DOCTYPE html>
      <meta charset="utf-8">
      <style>

        .links line {
          stroke: #999;
          stroke-opacity: 0.6;
        }

        .node text {
          pointer-events: none;
          font: 10px sans-serif;
        }

        .nodes circle {
          stroke: #fff;
          stroke-width: 1.5px;
        }

      </style>
      <svg width="960" height="600"></svg>
      <script type=text/javascript>
    """

  if supportsHTML
    # mbostock comment: https://github.com/mpld3/mpld3/pull/34
    d3HTML = d3HTML *
      """
        require.config({
          paths: {
            d3: '//d3js.org/d3.v4.min',
            mathjax: '//edx-static.s3.amazonaws.com/mathjax-MathJax-727332c/MathJax.js?config=TeX-MML-AM_HTMLorMML-full'
          },
          shim: {
            'd3': {
              exports: 'd3',
              init: function() {
                window.d3 = d3;
              }
            },
          }
        });
        require(["d3", "mathjax"], function(d3, mathjax) {
      """
  end

  d3HTML = d3HTML *
    """
      var svg = d3.select("svg"),
      width = +svg.attr("width"),
      height = +svg.attr("height");

      var color = d3.scaleOrdinal(d3.schemeCategory20);
      var simulation = d3.forceSimulation()
                          .force("link", d3.forceLink().id(function(d) { return d.id; }))
                          .force("charge", d3.forceManyBody())
                          .force("center", d3.forceCenter(width / 2, height / 2));

      var graph = $(json(data));
      var link = svg.append("g")
                    .attr("class", "links")
                    .selectAll("line")
                    .data(graph.links)
                    .enter().append("line")
                    .attr("stroke-width", function(d) { return Math.sqrt(d.value); });
      var node = svg.append("g")
                    .attr("class", "nodes")
                    .selectAll("circle")
                    .data(graph.nodes)
                    .enter().append("circle")
                    .attr("r", 5)
                    .attr("fill", function(d) { return color(d.group); })
                    .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended));

      node.append("title")
        .text(function(d) { return JSON.stringify(d.metadata); });
      node.append("text")
        .attr("dx", 12)
        .attr("dy", ".35em")
        .text(function(d) { return d.name });


      simulation
        .nodes(graph.nodes)
        .on("tick", ticked);

      simulation.force("link")
        .links(graph.links);

      function ticked() {
        link
          .attr("x1", function(d) { return d.source.x; })
          .attr("y1", function(d) { return d.source.y; })
          .attr("x2", function(d) { return d.target.x; })
          .attr("y2", function(d) { return d.target.y; });
        node
          .attr("cx", function(d) { return d.x; })
          .attr("cy", function(d) { return d.y; });
      }

      function dragstarted(d) {
        if (!d3.event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      }

      function dragged(d) {
        d.fx = d3.event.x;
        d.fy = d3.event.y;
      }

      function dragended(d) {
        if (!d3.event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      }
    """

  if supportsHTML
    d3HTML = d3HTML *
      """
        });
      """
  end

  d3HTML = d3HTML *
    """
      </script>
    """

  return d3HTML
end

# D3 operations on graph metadata
function getD3GraphMetadataHTML(graph::AD, supportsHTML::Bool)
  d3Data = getGraphMetadata(graph)
  println(json(d3Data))
  d3HTML = d3ForcedDirectedGraphHTML(d3Data, supportsHTML)

  #=
  displayid = "AutoDiff-iframe-" * string(rand())
  height = 500
  graphHTML =
    """
      <!DOCTYPE html>
      <body>
        <iframe id="$(displayid)" style="height:$(height)px;width=100%" src="data:text/html;charset=utf-8,$(d3HTML)"/>
      </body>
    """
  =#

  return d3HTML
end

# display in a new window (Electron) in case
# text/html mime type is not supported
function blink_show(graph::AD)
    w = Window()
    try
      d3 = joinpath(dirname(@__FILE__), "..", "deps", "d3.min.js")
      for file in (d3,)
        load!(w, file)
      end
    catch
      error("Could not load the file: d3.min.js from the deps directory.")
    end
    body = getD3GraphMetadataHTML(graph, false)
    body!(w, body)
end

# plot the graph
function plot(graph::AD)
  if displayable("text/html")
    display("text/html", getD3GraphMetadataHTML(graph, true))
  else
    #try
      blink_show(graph)
    #catch
    #  error("MIME type text/html is not available.")
    #end
  end
end
