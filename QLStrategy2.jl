

type QLStrategy2 <: Strategy
    nn::NN
    actions::Array{QLAction,1}
end

QLStrategy2(nn::NN) = QLStrategy2(nn, Array{QLAction, 1}())

function construct_QL2()
    nn = NN()
    nn.batch_size = N_BATCH
    add_layer(nn, RectifiedLayer(N_INPUT_LAYER, N_HIDDEN_LAYER))
    add_layer(nn, RectifiedLayer(N_HIDDEN_LAYER, N_HIDDEN_LAYER2))
    add_layer(nn, RectifiedLayer(N_HIDDEN_LAYER2, N_HIDDEN_LAYER3))
    add_layer(nn, RectifiedLayer(N_HIDDEN_LAYER3, N_HIDDEN_LAYER4))
    add_layer(nn, RectifiedLayer(N_HIDDEN_LAYER4, N_HIDDEN_LAYER5))
    add_layer(nn, LinearLayer(N_HIDDEN_LAYER5, N_OUTPUT_LAYER))
    return nn
end


function new_hand(strategy::QLStrategy2)
    strategy.actions = Array{QLAction, 1}()
end

function get_action(strategy::QLStrategy2, b::Board)
    #action = NN_ACTION(7 - Base.convert(Int, floor(log2(1.0+rand()*127))))
    stack = b.stacks[1]
    fold_value = -b.contributed
    inputs = encode_board(b, strategy)
    outputs = feedforward(strategy.nn, inputs)

    estimates = outputs[end]
    m = maximum(estimates)
    if isnan(m)
        println("NAN!!!!")
        exit()
    end
    action = QL_ACTION(findfirst(estimates, m))
    if rand() < EXPLOITATION
        action = QL_ACTION(Int(floor(rand() * 7 + 1)))
    end

    if fold_value > m
        action = A_FOLD
        if b.to_call == 0
            action = A_CHECKCALL
        end
    end
 
     if Int(action) > Int(A_CHECKCALL) && b.raises >= MAX_RAISES
        action = A_CHECKCALL
    end
    
    a=convert_ql_action(b.big_blind, action)
    while ((a.amount + b.to_call) > stack) && (Int(action) > 1)
        action = QL_ACTION(Int(action) - 1)
        a=convert_ql_action(b.big_blind, action)
    end
    push!(strategy.actions, QLAction(action, inputs, estimates))
   
    #println(a)
    return a
end

function is_log_space(strategy::QLStrategy2)
    return true
end

function hand_finished(strategy::QLStrategy2,  result::Int, growth::Float, would_have_won::Bool, stack_before::Int)
    delta = 0.0
    r = 0.0
    if length(strategy.actions) == 0
        return 0.0
    end
    for action in strategy.actions
         if action.a == A_FOLD
            continue
         end
         target = fill(Float(safeLog(growth)), length(action.estimates))
         add_experience!(strategy, target, action.inputs, action.a)
    end
    return result
end




