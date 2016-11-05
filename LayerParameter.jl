typealias ParameterVector Vector{Float}

type LayerParameter
    values::ParameterVector
    gradient_sum::ParameterVector
    last_gradient_sum::ParameterVector
    learning_rates::ParameterVector
    rolling_average::ParameterVector
    derivation_function::Function
    special_backprop_function::Function
end

disable_normal_parameter_updates = function(f::OutputVector)
    return 0.0
end

dummy_special_backprop_function = function(parameter::LayerParameter, deltas::DeltaVector, outputs::OutputVector)
    return nothing
end

function parameter_special_backprop(parameter::LayerParameter, deltas::DeltaMatrix, outputs::OutputMatrix)
    if parameter.special_backprop_function == dummy_special_backprop_function
        return nothing
    end
    rows, columns = size(outputs)
    for i in 1:rows
        parameter_special_backprop(parameter, vec(deltas[i, :]), vec(outputs[i, :]))
    end
end

function parameter_special_backprop(parameter::LayerParameter, deltas::DeltaVector, outputs::OutputVector)
    parameter.special_backprop_function(parameter, deltas, outputs)
end




