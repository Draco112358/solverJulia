using Genie, Genie.Renderer, Genie.Renderer.Html, Genie.Renderer.Json, Genie.Requests
include("solve.jl")

route("/") do
  html("Hello World")
end

route("/solving", method="POST") do 
  return doSolving(jsonpayload()["mesherOutput"], jsonpayload()["solverInput"], jsonpayload()["solverAlgoParams"])
end

up(8001, async = false)