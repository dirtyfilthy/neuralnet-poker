

type SparseEncoder <: PerceptronLayer
    weights::Matrix{Float64}
    n_inputs::Int
    n_outputs::Int
    k::Int
    bias_learningrate::Float64
end

function SparseEncoder(n_inputs::Int, n_outputs::Int, k::Int, bias_learningrate::Float64)
    s = SparseEncoder(rand(n_inputs + 1, n_outputs) * 2.0 - 1.0, n_inputs, n_outputs, k, bias_learningrate)
    s.weights[:,1] = zeroes(s[:,1])
    return s
end

function is_output_activated(layer::SparseEncoder, outputs::Vector{Float64})
    activated = falses(outputs)
    temp = copy(outputs)
    for i = 1:layer.k
        m = max(temp)
        index = findfirst(temp, m)
        activated[index] = true
        temp[index] = 0.0
    end
    return activated
end

function process_raw_output(outputs::Vector{Float64}, activated::Vector{Bool})
    result = zeroes(outputs)
    for i = 1:length(outputs)
        if activated[i]
            result[i] = outputs[i]
        end
    end
    return result
end

function output(layer::SparseEncoder, inputs::Vector{Float64})
    output = layer.weights * inputs_with_bias(layer, inputs)' 
    activated = is_output_activated(layer, output)
    return process_raw_output(output, activated)
end

function update!(layer::SparseEncoder, inputs::Vector{Float64}, deltas::Vector{Float64}, learningrate::Float64)
    if length(deltas) != layer.n_outputs
        error("Expected $(layer.n_outputs) outputs, got $(length(training_outputs))")
    end
    output = layer.weights * inputs_with_bias(layer, inputs)' 
    activated = is_output_activated(layer, output)
    gradients = processed_inputs(layer, inputs)' * deltas
    bias = layer.weights[:, 1]
    k_ratio = layer.n_outputs / (k + 0.0) 
    for i = 1:bias
        z = 0.0
        if activated[i]
            z = 1.0
        end
        bias[i] = bias[i] + (layer.bias_learningrate * (k_ratio - z))
    end
    layer.weights = layer.weights - (gradients * learningrate)
    layer.weights[:, 1] = bias

end