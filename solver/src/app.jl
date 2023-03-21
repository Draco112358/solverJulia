using Genie, Genie.Renderer, Genie.Renderer.Html

route("/") do
  html("Hello World")
end


up(8001, async = false)