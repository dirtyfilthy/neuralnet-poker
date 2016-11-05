const USE_RMSPROP = false
const RMSPROP_STEPRATE = 0.0001
const RMSPROP_EPSILON = 0.0000001
const RMSPROP_DECAY = 0.95
const RMSPROP_BATCH = 500

const RPROP_LOWER_BOUND = 0.00001
const RPROP_UPPER_BOUND = 0.05
const RPROP_UP = 1.1
const RPROP_DOWN = 0.5

type NN
    layers::Array{NNLayer,1}
    batch_size::Int 
    iteration::Int
    last_error::Float64
    error::Float64
end



NN() = NN(Array{NNLayer,1}(), 1, 0, 9999999999.0, 0.0)

function add_layer(nn::NN, layer::NNLayer)
    if length(nn.layers) > 0
        top_outputs = nn.layers[end].n_outputs
        if top_outputs != layer.n_inputs
            error("Inputs of new layer ($(layer.n_inputs)) do not match outputs of old layer ($top_outputs)")
        end
    end
    push!(nn.layers, layer)
end

n_outputs(nn::NN) = nn.layers[end].n_outputs
n_inputs(nn::NN) = nn.layers[1].n_inputs


with_bias(inputs::Vector{Float}) = vcat(inputs, [1.0])

function feedforward(nn::NN, inputs::InputMatrix)
    outputs = OutputMatrixList()
    findEvilNan("feedforward inputs", inputs)
    push!(outputs, inputs')
    i=0
    #println("feedforward")
    for layer in nn.layers
        i=i+1
        findEvilNan("feedforward weights layer $i", layer.weights)
        o = output(layer, inputs)
        findEvilNan("in feedforward outputs o $i", o)
        inputs = o'
        push!(outputs, o)
    end
    return outputs
end

function feedforward(nn::NN, inputs::InputVector)
    outputs = OutputList()
    push!(outputs, inputs)
    if length(inputs) != nn.layers[1].n_inputs
        error("Expected $(nn.layers[1].n_inputs) inputs but got $(length(inputs))")
    end
    i = 0
    for layer in nn.layers
        i += 1
        o = output(layer, inputs)
        findEvilNan("in feedforward outputs layer $i", o)
        inputs = o
        push!(outputs, o)
    end
    return outputs
end

output(nn::NN, inputs::InputVector) = feedforward(nn, inputs)[end]


function calc_deltas(nn::NN, outputs::OutputList, initial_deltas::DeltaVector)
    n_layers = length(nn.layers)
    
    deltas = OutputList(n_layers)
    final_outputs = outputs[end]
    deltas[end] = derive(nn.layers[end], final_outputs) .* initial_deltas
    if n_layers == 1
        return deltas
    end
    for i in 1:n_layers-1
        current_idx = n_layers - i
        higher_idx = n_layers - i + 1
        current_output = outputs[end - i]
        current_layer = nn.layers[current_idx]
        higher_layer = nn.layers[higher_idx]

        d = derive(current_layer, current_output) 
        @fastmath v = vec(higher_layer.weights * deltas[higher_idx])
        @fastmath deltas[current_idx] = d .* v

    end
    return deltas
end

function calc_deltas(nn::NN, outputs::OutputMatrixList, initial_deltas::DeltaMatrix)
    n_layers = length(nn.layers)
    
    deltas = OutputMatrixList(n_layers)
    final_outputs = outputs[end]
    findEvilNan("calc_deltas  in final outputs", final_outputs)
    output_gradient = derive(nn.layers[end], final_outputs) 
    findEvilNan("calc_deltas derive in final outputs", output_gradient)
    deltas[end] = output_gradient .* initial_deltas
    if n_layers < 2
        return deltas
    end
    for i in 1:n_layers-1
        current_idx = n_layers - i
        higher_idx = n_layers - i + 1
        current_output = outputs[end - i]
        current_layer = nn.layers[current_idx]
        higher_layer = nn.layers[higher_idx]

        d = derive(current_layer, current_output)
        findEvilNan("calc_deltas derive in layer $current_idx", d)
        @fastmath v = deltas[higher_idx] * higher_layer.weights'
        @fastmath deltas[current_idx] = d .* v

    end
    return deltas
end

function backpropagate!(nn::NN, deltas::OutputList, outputs::OutputList)
    n_layers = length(nn.layers)
    for i in 0:n_layers-1
        current_idx = n_layers - i
        current_layer = nn.layers[current_idx]
        inputs = outputs[current_idx]
        #findEvilNan("backpropagate inputs layer $current_idx" ,inputs)
        gradients = inputs * deltas[current_idx]'
        #findEvilNan("backpropagate deltas layer $current_idx" ,deltas[current_idx])
        bias_gradients = deltas[current_idx]
        raw_outputs = z(current_layer, inputs)
        
        @fastmath current_layer.gradient_sum += gradients
        @fastmath current_layer.bias_gradient_sum += bias_gradients
        backpropagate_layer_parameters!(current_layer, deltas[current_idx], raw_outputs)
    end
end

function backpropagate!(nn::NN, deltas::Vector{Float}, outputs::OutputList)
    findEvilNan("backpropagate initial deltas", deltas)
    deltas = calc_deltas(nn, outputs, deltas)
    backpropagate!(nn, deltas, outputs)
   
end

function backpropagate!(nn::NN, deltas::DeltaMatrix, outputs::OutputMatrixList)
    deltas = calc_deltas(nn, outputs, deltas)

    
    n_layers = length(nn.layers)
    for i in 0:n_layers-1

        current_idx = n_layers - i
        current_layer = nn.layers[current_idx]
        inputs = outputs[current_idx]
        findEvilNan("backpropagate inputs layer $current_idx" ,inputs)
        gradients = inputs' * deltas[current_idx]
        findEvilNan("backpropagate deltas layer $current_idx" ,deltas[current_idx])
        findEvilNan("backpropagate gradients $current_idx" , gradients)
        rows, columns = size(deltas[current_idx])
        bias_gradients = fill(1.0, 1, rows) * deltas[current_idx]
        raw_outputs = z(current_layer, inputs')
        
        @fastmath current_layer.gradient_sum += gradients
        @fastmath current_layer.bias_gradient_sum += vec(bias_gradients)
        backpropagate_layer_parameters!(current_layer, deltas[current_idx], raw_outputs)
    end
end


function update!(nn::NN)
   

    for current_layer in nn.layers

        sum_gradients = current_layer.gradient_sum
        sum_bias = current_layer.bias_gradient_sum
        weight_updates = nothing
        bias_updates = nothing

        # calculate learning rate update 

        @fastmath sign_change = map(x -> x < 0, current_layer.last_gradient .* sum_gradients)
        @fastmath bias_sign_change = map(x -> x < 0, current_layer.bias_last_gradient .* sum_bias)
        @fastmath bigger = min(current_layer.learning_rates * RPROP_UP, RPROP_UPPER_BOUND)
        @fastmath smaller = max(current_layer.learning_rates * RPROP_DOWN, RPROP_LOWER_BOUND)
        @fastmath bias_bigger = min(current_layer.bias_learning_rates * RPROP_UP, RPROP_UPPER_BOUND)
        @fastmath bias_smaller = max(current_layer.bias_learning_rates * RPROP_DOWN, RPROP_LOWER_BOUND)
        @fastmath current_layer.learning_rates = ifelse(sign_change, smaller, bigger)
        @fastmath current_layer.bias_learning_rates = ifelse(bias_sign_change, bias_smaller, bias_bigger)

      
        if USE_RMSPROP    
            @fastmath weight_updates = (current_layer.learning_rates ./ ((current_layer.rolling_average ^ 0.5) + RMSPROP_EPSILON)) .* (sum_gradients ./ RMSPROP_BATCH)
            @fastmath bias_updates = (current_layer.bias_learning_rates ./ ((current_layer.bias_rolling_average ^ 0.5) + RMSPROP_EPSILON)) .* (sum_bias ./ RMSPROP_BATCH)
            @fastmath current_layer.rolling_average = (RMSPROP_DECAY * current_layer.rolling_average) .+ ((1.0 - RMSPROP_DECAY) * ((sum_gradients ./ RMSPROP_BATCH).^ 2))
            @fastmath current_layer.bias_rolling_average = (RMSPROP_DECAY * current_layer.bias_rolling_average) .+ ((1.0 - RMSPROP_DECAY) * ((sum_bias  ./ RMSPROP_BATCH) .^ 2))
            findEvilNan("rolling avergage", current_layer.rolling_average)
        else
           
            @fastmath bias_updates     = (sign(sum_bias)       .* current_layer.bias_learning_rates)
            @fastmath weight_updates   = (sign(sum_gradients)  .* current_layer.learning_rates)
        end
        
        findEvilNan("sum gradients", sum_gradients)
        
        
        # calculate weight update 

       
        #println("rolling $(current_layer.rolling_average )")
        
        
        findEvilNan("weight updates", weight_updates)
        #@fastmath weight_updates = (sign(sum_gradients)  .* current_layer.learning_rates)
       

        # apply updates

        current_layer.weights = current_layer.weights - weight_updates
        current_layer.biases  = current_layer.biases  - bias_updates

        # refresh for next time

        current_layer.last_gradient = sum_gradients
        current_layer.bias_last_gradient = sum_bias                                     
        current_layer.gradient_sum = zeros(sum_gradients)
        current_layer.bias_gradient_sum = zeros(sum_bias)
        update_layer_parameters!(current_layer)
    end
            
end
