abstract NNLayer

function output(layer::NNLayer, inputs::InputVector)
    zed::OutputVector = z(layer, inputs)
    act::OutputVector = activate(layer, zed)
    return act
end

function output(layer::NNLayer, inputs::InputMatrix) 
    findEvilNan("output weights", layer.weights)
    findEvilNan("output inputs", inputs)
    outputs::OutputMatrix = z(layer, inputs)
    findEvilNan("output outputs", outputs)
    rows, columns = size(outputs)
    for i in 1:rows
        act::OutputVector = activate(layer, vec(outputs[i, :]))
        findEvilNan("output act", act)
        outputs[i, :] = act
    end
    return outputs
end


z(layer::NNLayer, inputs::InputVector) = vec(inputs' * layer.weights) + layer.biases

function z(layer::NNLayer, inputs::InputMatrix)
    findEvilNan("z weights", layer.weights)
    findEvilNan("z biases", layer.biases)
    return broadcast(+, layer.biases', inputs' * layer.weights)
end

activate(layer::NNLayer, outputs::OutputVector) = layer.activation_function(outputs)

derive(layer::NNLayer, outputs::OutputVector) = layer.derivation_function(outputs)

function derive!(layer::NNLayer, outputs::OutputMatrix)
    rows, columns = size(outputs)
    for i in 1:rows
        outputs[i, :] = derive(layer, vec(outputs[i, :]))
    end
    return outputs
end

derive(layer::NNLayer, outputs::OutputMatrix) = derive!(layer, copy(outputs))


function add_layer_parameter!(layer::NNLayer, derivation_function::Function; special_function::Function = dummy_special_backprop_function)
    values = fillfloat(0.0, layer.n_outputs)
    gradient_sum = zeros(values)
    last_gradient_sum = zeros(values)
    learning_rates = fillfloat(0.01, length(values))
    rolling_average = fillfloat(0.0, length(values))
    lp = LayerParameter(values, gradient_sum, last_gradient_sum, learning_rates, rolling_average, derivation_function, special_function)
    push!(layer.layer_parameters, lp)
    return lp
end

function backpropagate_layer_parameters!(layer::NNLayer, deltas::DeltaVector, outputs::OutputVector)
    for parameter in layer.layer_parameters
        updates = deltas .* parameter.derivation_function(outputs)
        parameter.gradient_sum += updates
        parameter_special_backprop(parameter, deltas, outputs)
    end
end

function backpropagate_layer_parameters!(layer::NNLayer, deltas::DeltaMatrix, outputs::OutputMatrix)
    o = copy(outputs)
    for parameter in layer.layer_parameters
        rows, columns = size(outputs)
        for i in 1:rows
            o[i, :] = parameter.derivation_function(vec(o[i, :]))
        end
        updates = deltas .* o
     
        rows, columns = size(updates)
        parameter.gradient_sum += vec(fillfloat(1.0, 1, rows) * updates)
        parameter_special_backprop(parameter, deltas, outputs)
    end
end


function update_layer_parameters!(layer::NNLayer)
    for parameter in layer.layer_parameters
        weight_update = nothing
        @fastmath sign_change = map(x -> x < 0, parameter.last_gradient_sum .* parameter.gradient_sum)
        @fastmath bigger = min(parameter.learning_rates * RPROP_UP, RPROP_UPPER_BOUND)
        @fastmath smaller = max(parameter.learning_rates * RPROP_DOWN, RPROP_LOWER_BOUND)
        @fastmath parameter.learning_rates = ifelse(sign_change, smaller, bigger)

        if !USE_RMSPROP  
            @fastmath weight_updates = (sign(parameter.gradient_sum)  .* parameter.learning_rates)
        else
            
            @fastmath weight_updates = (parameter.learning_rates ./ ((parameter.rolling_average ^ 0.5) + RMSPROP_EPSILON)) .* 
                        (parameter.gradient_sum ./ RMSPROP_BATCH)
            @fastmath parameter.rolling_average = (RMSPROP_DECAY * parameter.rolling_average) + 
            ((1.0 - RMSPROP_DECAY) * ((parameter.gradient_sum ./ RMSPROP_BATCH) .^ 2))
        end
        
        @fastmath parameter.values -= weight_updates

        parameter.last_gradient_sum = parameter.gradient_sum
        parameter.gradient_sum = zeros(parameter.last_gradient_sum)
    end
end

