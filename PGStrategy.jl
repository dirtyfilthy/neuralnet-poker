@enum PG_ACTION PG_FOLD=1 PG_CHECKCALL=2 PG_RAISEBB=3 PG_RAISE2BB=4 PG_RAISE4BB=5 PG_RAISE8BB=6 PG_RAISE16BB=7 PG_RAISE32BB=8

PG_OUTPUT_LAYER = 8
PG_HIDDEN_LAYER = 120
PG_HIDDEN_LAYER2 = 80
PG_HIDDEN_LAYER3 = 60
PG_HIDDEN_LAYER4 = 40
PG_HIDDEN_LAYER5 = 20
PG_EXPLOIT = 1.00

type PGStrategy <: Strategy
    nn::NN
    experiences::Vector{Experience}
    reward::Int
end

PGStrategy(nn::NN) = PGStrategy(nn, Vector{Experience}, 0)


function convert_pg_action(bb::Int, action::PG_ACTION)
    if action == PG_FOLD
        return Action(fold, 0)
    elseif action == PG_CHECKCALL
        return Action(call, 0)
    elseif action == PG_RAISEBB
        return Action(raise, bb)
    elseif action == PG_RAISE2BB
        return Action(raise, 2*bb)
    elseif action == PG_RAISE4BB
        return Action(raise, 4*bb)
    elseif action == PG_RAISE8BB
        return Action(raise, 8*bb)
    elseif action == PG_RAISE16BB
        return Action(raise, 16*bb)
    else action == PG_RAISE32BB
        return Action(raise, 32*bb)
    end
end

function construct_PG()
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

function new_game(strategy::PGStrategy)
    strategy.experiences = Vector{Experiences}()

end



function new_hand(strategy::PGStrategy)
    strategy.hands += 1
end

function hand_finished(strategy::PGStrategy, result::Int, growth::Float, would_have_won::Bool, stack_before::Int)
    
    if length(strategy.actions) == 0
        return 0.0
    end
     if isnan(growth) || growth == 0.0 
        growth = 0.000000000001
    end

    strategy.reward += log(growth)
    return r
end

function is_log_space(strategy::PGStrategy)
    return true
end

function game_finished(strategy::PGStrategy)
    r = strategy.reward
 
    if strategy.hands == 0
        return nothing
    end
    for experience in strategy.experiences
        experience.deltas *= r
    end
    add_experience!(strategy, strategy.experiences)
   
    
    #strategy.exploit = strategy.exploit * 0.999
    
end
    



function get_action(strategy::PGStrategy, b::Board)
    #action = NN_ACTION(7 - Base.convert(Int, floor(log2(1.0+rand()*127))))
    stack = b.stacks[1]
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
    #if rand() < strategy.exploit
    #    action = PG_ACTION(Int(floor(rand() * 7 + 2)))
    #end
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
    probs[actionint] -= 1.0
    push!(strategy.experiences, GenericExperience(probs, outputs))
    return a
end