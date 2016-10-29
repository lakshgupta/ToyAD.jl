using BinDeps
@BinDeps.setup

# download https:////d3js.org/d3.v4.min.js to deps/ if it doesn't exist
local_fn = joinpath(dirname(@__FILE__), "d3.min.js")
if !isfile(local_fn)
	info("Cannot find deps/d3.min.js... downloading latest version.")
	download("https://d3js.org/d3.v4.min.js", local_fn)
end

@BinDeps.install