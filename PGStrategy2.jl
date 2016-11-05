
PG2_DISCOUNT = 0.98

type PG2BoardProb
    prob::Vector{Float64}
    stack::Int
    outputs::OutputList
end

type ActionList 
    actions::Vector{PG2BoardProb}
    reward::Float64
end

type PGStrategy2 <: Strategy
    nn::NN
    hand_list::Vector{ActionList}
    stack::Int
end

ActionList() = ActionList(Vector{PG2BoardProb}(), 0.0)



PGStrategy2(nn::NN) = PGStrategy2(nn, Vector{ActionList}(), Vector{Float64}(), -999)


function construct_PG2()
    nn = NN()
    nn.batch_size = 250
    add_layer(nn, RectifiedLayer(N_INPUT_LAYER, PG_HIDDEN_LAYER))
    add_layer(nn, RectifiedLayer(PG_HIDDEN_LAYER, PG_HIDDEN_LAYER2))
    add_layer(nn, RectifiedLayer(PG_HIDDEN_LAYER2, PG_HIDDEN_LAYER3))
    add_layer(nn, RectifiedLayer(PG_HIDDEN_LAYER3, PG_HIDDEN_LAYER4))
    add_layer(nn, RectifiedLayer(PG_HIDDEN_LAYER4, PG_HIDDEN_LAYER5))
    add_layer(nn, SoftMaxLayer(PG_HIDDEN_LAYER5, PG_OUTPUT_LAYER))
    return nn
end

function new_game(strategy::PGStrategy2)
    strategy.hand_list = Vector{ActionList}()
    strategy.stack = -999

end



function new_hand(strategy::PGStrategy2)
    push!(strategy.hand_list, ActionList())
end


function hand_finished(strategy::PGStrategy2, result::Int, growth::Float, would_have_won::Bool, stack_before::Int)
    
    if length(strategy.hand_list[end].actions) == 0
        return 0.0
    end
    if isnan(growth) || growth == 0.0 
        growth = 0.000000000001
    end
    r = log(growth)
    hand = strategy.hand_list[end]
    strategy.stack += result
    hand.reward = r
    return r
end

function is_log_space(strategy::PGStrategy2)
    return true
end

function game_finished(strategy::PGStrategy2)
 
   

    if length(strategy.hand_list) == 0
        return nothing
    end
    reward_list = Vector{Float64}()
    running = 0.0
    hands = reverse(strategy.hand_list)
    for hand in hands
        reward = hand.reward + running
        running = reward * 0.99
        for action in hand.actions
            growth = strategy.stack / action.stack
            if isnan(growth) || growth <= 0.0
                growth = 0.0000001
            end
            reward = log(growth)
            add_experience!(strategy, hand.probs * reward, hand.outputs)
        end
    end;

end



function get_action(strategy::PGStrategy2, b::Board)
    #action = QL_ACTION(7 - Base.convert(Int, floor(log2(1.0+rand()*127))))
    stack = b.stacks[1]
    if strategy.stack == -999
        strategy.stack = stack
    end
    fold_value = -b.contributed
    outputs = feedforward(strategy.nn, encode_board(b, strategy))
    probs = outputs[end]
    c = 0.0
    r = rand()
    action = PG_FOLD
    for i in 1:length(probs)
        c += probs[i]
        if r < c
            action = PG_ACTION(i)
            break
        end
    end
    if action == PG_FOLD && b.to_call == 0
        action = PG_CHECKCALL
    end
    if Int(action) > Int(PG_CHECKCALL) && b.raises >= MAX_RAISES
        action = PG_CHECKCALL
    end
    a = convert_pg_action(b.big_blind, action)
    while a.amount + b.to_call > stack && Int(action) > 2
        action = PG_ACTION(Int(action) - 1)
        a=convert_pg_action(b.big_blind, action)
    end
    actionint = Int(action)
    probs[actionint] -= 1
  
    #println("action $action probs $probs")
    push!(hand.actions, PG2BoardProb(probs, stack, outputs))
    return a
end