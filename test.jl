include("Simulation.jl")

nn = Simulation.NN()
Simulation.add_layer(nn, Simulation.TanhLayer(4, 2))
Simulation.add_layer(nn, Simulation.LinearLayer(2, 1))

for(i in 1:1000000)
    inputs = rand(4)
    t = sum(inputs)
    outputs = Simulation.feedforward(nn, inputs)
    o = outputs[end][1]
    delta = o - t
    println("output $o target $t delta $delta")
    Simulation.backpropagate(nn, [delta], outputs, 0.01)
end