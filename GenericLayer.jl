const LEARNING_RATE_START = 0.00001

type GenericLayer <: NNLayer
    weights::Matrix{Float}
    last_weight_update::Matrix{Float}
    last_bias_update::Vector{Float}
    n_inputs::Int
    n_outputs::Int
    last_gradient::Matrix{Float}
    bias_last_gradient::Vector{Float}
    gradient_sum::Matrix{Float}
    bias_gradient_sum::Vector{Float}
    learning_rates::Matrix{Float}
    bias_learning_rates::Vector{Float}
    biases::Vector{Float}
    activation_function::Function
    derivation_function::Function
    layer_parameters::Vector{LayerParameter}
    rolling_average::Matrix{Float}
    bias_rolling_average::Vector{Float} 

end

function GenericLayer(n_inputs::Int, n_outputs::Int, activation::Function, derivation::Function)
    r_matrix = ((randfloat(n_inputs, n_outputs)*2.0) - 1.0) * (sqrt(6) / sqrt(n_inputs + n_outputs))
    b = fillfloat(0.0, n_outputs)
    rolling = fill(1000.0, n_inputs, n_outputs)
    bias_rolling= fill(1000.0, n_outputs)
    return GenericLayer(r_matrix, zeros(r_matrix), zeros(b), n_inputs, n_outputs, zeros(r_matrix), copy(b), zeros(r_matrix), copy(b),
        fillfloat(LEARNING_RATE_START, n_inputs, n_outputs), fillfloat(LEARNING_RATE_START, n_outputs), b, activation, derivation, 
        Vector{LayerParameter}(), rolling, bias_rolling)
end


function PReLULayer(n_inputs::Int, n_outputs::Int)
    a = nothing
    activation = function(f::OutputVector)
        with_a = f .* a.values
        return ifelse(f .< 0.0, with_a, f)
    end

    derivation = function(f::OutputVector)
        return ifelse(f .< 0.0, a.values, ones(f))
    end

    parameter_derivation = function(f::OutputVector)
        return ifelse(f .< 0.0, f, zeros(f))
    end

    layer = GenericLayer(n_inputs, n_outputs, activation, derivation)
    a = add_layer_parameter!(layer, parameter_derivation)

    return layer

end

function linear_activation_function(f::OutputVector)
    return f
end

function linear_derivation_function(f::OutputVector)
    ones(f)
end



LinearLayer(n_inputs::Int, n_outputs::Int) = GenericLayer(n_inputs, n_outputs, linear_activation_function, linear_derivation_function)

SoftPlusLayer(n_inputs::Int, n_outputs::Int) = GenericLayer(n_inputs, n_outputs, f->map(x->log(Float(1.0) + exp(Float(x))),f),
 f->map(x->Float(1.0) / (Float(1.0) + exp(Float(-x))),f))


function KSparseLayer(n_inputs::Int, n_outputs::Int; with_bias = false, k=0)
    layer = nothing
    let b = nothing
        activation = nothing
        if k == 0
            k = n_outputs / 2
        end
        let k = k
            activation = function(f)
                sorted = sort(f, reverse=true)
                cutoff = f[k]
                r = ifelse((f + b.values) .< cutoff, zeros(f), f + b.values)
                return r
            end
        end
        opts = Dict{String, Any}()
  

        derivation = function(f)
            return map(x->x == 0.0f0 ? 0.0f0 : 1.0f0)
        end

  
        special_backprop = function(parameter::LayerParameter, deltas::DeltaVector, outputs::OutputVector)
            bigger  = parameter.gradient_sum + 0.01
            smaller = parameter.gradient_sum - 0.01
            parameter.gradient_sum  = ifelse(deltas .== 0.0, smaller, bigger)
        end

        layer = GenericLayer(n_inputs, n_outputs, activation, derivation)
        if with_bias
            b = add_layer_parameter!(layer, disable_normal_parameter_updates, special_function = special_backprop)
        end
    end
    return layer
end

function SoftMaxLayer(n_inputs::Int, n_outputs::Int)
    activation = function(f)
        big = maximum(f)
        m = exp(f - big)
        r = m / sum(m)
        return r
    end

    derivation = function(f)
        r = map(x->(1.0 - x)*x, f)
        return r
    end
    return GenericLayer(n_inputs, n_outputs, activation, derivation)
end

RectifiedLayer(n_inputs::Int, n_outputs::Int) = GenericLayer(n_inputs, n_outputs, f->map(x->max(0.0, x), f),
 f->map(x->x < 0.0 ? 0.0 : 1.0, f))

LeakyRELULayer(n_inputs::Int, n_outputs::Int) = GenericLayer(n_inputs, n_outputs, f->map(x->x < 0 ? 0.01*x : x, f),
 f->map(x->x < 0.0 ? 0.01 : 1.0, f))



TanhLayer(n_inputs::Int, n_outputs::Int) = GenericLayer(n_inputs, n_outputs, f->map(x->tanh(x), f), f->map(x->1.0 - (x ^ 2), f))
