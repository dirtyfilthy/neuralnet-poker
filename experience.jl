abstract Experience

type GenericExperience <: Experience 
    targets::Vector{Float}
    inputs::InputVector
    action::Any
end

targets(e::Experience) = e.targets
inputs(e::Experience) = e.inputs

function output(nn::NN, e::Experience)
    i=inputs(e)
    return feedforward(nn, i)
end

function deltas(nn::NN, e::Experience, outputs::OutputVector) 
    d = zeros(outputs)
    actionint = Int(e.action)
    d[actionint] = outputs[actionint] - e.targets[actionint]
    return d
end

surprise(e::Experience)=e.surprise

