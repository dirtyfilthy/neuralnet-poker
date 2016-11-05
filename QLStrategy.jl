@enum QL_ACTION A_FOLD=0 A_CHECKCALL=1 A_RAISEBB=2 A_RAISE2BB=3 A_RAISE4BB=4 A_RAISE8BB=5 A_RAISE16BB=6 A_RAISE32BB=7


N_OUTPUT_LAYER = 7
N_HIDDEN_LAYER = 120
N_HIDDEN_LAYER2 = 80
N_HIDDEN_LAYER3 = 60
N_HIDDEN_LAYER4 = 40
N_HIDDEN_LAYER5 = 20
N_BATCH = 250
EXPLOITATION = 0.01

type QLAction
    a::QL_ACTION
    inputs::Vector{Float}
    estimates::Vector{Float}
end


type QLStrategy <: Strategy
    nn::NN
    actions::Array{QLAction,1}
end

QLStrategy(nn::NN) = QLStrategy(nn, Array{QLAction, 1}())

function construct_QL()
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

function convert_ql_action(bb::Int, action::QL_ACTION)
    if action == A_FOLD
        return Action(fold, 0)
    elseif action == A_CHECKCALL
        return Action(call, 0)
    elseif action == A_RAISEBB
        return Action(raise, bb)
    elseif action == A_RAISE2BB
        return Action(raise, 2*bb)
    elseif action == A_RAISE4BB
        return Action(raise, 4*bb)
    elseif action == A_RAISE8BB
        return Action(raise, 8*bb)
    elseif action == A_RAISE16BB
        return Action(raise, 16*bb)
    else action == A_RAISE32BB
        return Action(raise, 32*bb)
    end
end

function new_hand(strategy::QLStrategy)
    strategy.actions = Array{QLAction, 1}()
end

function get_action(strategy::QLStrategy, b::Board)
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

function is_log_space(strategy::QLStrategy)
    return false
end

function hand_finished(strategy::QLStrategy,  result::Int, growth::Float, would_have_won::Bool, stack_before::Int)
    delta = 0.0
    r = 0.0
    if length(strategy.actions) == 0
        return 0.0
    end
    for action in strategy.actions
         if action.a == A_FOLD
            continue
         end
         target = fill(profitToFloat(20, result), length(action.estimates))
         add_experience!(strategy, target, action.inputs, action.a)
    end
    return result
end




