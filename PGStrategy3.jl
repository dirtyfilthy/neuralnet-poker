
const PG3EXPLOIT=0.02

type PGStrategy3 <: Strategy
    nn::NN
    experiences::Vector{Experience}
end

type PG3Experience <: Experience 
    inputs::Vector{Float64}
    reward::Float64
    action::PG_ACTION
end

function deltas(nn::NN, e::PG3Experience, outputs::OutputVector)
    targets = copy(outputs)
    targets[Int(e.action)] -= 1.0
    return targets * e.reward
end



PGStrategy3(nn::NN) = PGStrategy3(nn, Vector{Experience}())


function construct_PG3()
    nn = NN()
    nn.batch_size = 250
    add_layer(nn, PReLULayer(N_INPUT_LAYER, PG_HIDDEN_LAYER))
    add_layer(nn, PReLULayer(PG_HIDDEN_LAYER, PG_HIDDEN_LAYER2))
    add_layer(nn, PReLULayer(PG_HIDDEN_LAYER2, PG_HIDDEN_LAYER3))
    add_layer(nn, PReLULayer(PG_HIDDEN_LAYER3, PG_HIDDEN_LAYER4))
    add_layer(nn, PReLULayer(PG_HIDDEN_LAYER4, PG_HIDDEN_LAYER5))
    add_layer(nn, SoftMaxLayer(PG_HIDDEN_LAYER5, PG_OUTPUT_LAYER))
    return nn
end

function new_game(strategy::PGStrategy3)

end



function new_hand(strategy::PGStrategy3)
    strategy.experiences = Vector{PG3Experience}()

end


function is_log_space(strategy::PGStrategy3)
    return true
end

function hand_finished(strategy::PGStrategy3, result::Int, growth::Float, would_have_won::Bool, stack_before::Int)
    reward = safeLog(growth)
    for experience in strategy.experiences
        experience.reward = reward
    end
    add_experience!(strategy, strategy.experiences)
    return result
end

function game_finished(strategy::PGStrategy3)
 

end



function get_action(strategy::PGStrategy3, b::Board)
    #action = QL_ACTION(7 - Base.convert(Int, floor(log2(1.0+rand()*127))))
    stack = b.stacks[1]
    fold_value = -b.contributed
    inputs = encode_board(b, strategy)
    outputs = feedforward(strategy.nn, inputs)
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
    if rand() < PG3EXPLOIT
        action = PG_ACTION(intRandFromOne(7)+1)
    end

    if action == PG_FOLD && b.to_call == 0
        action = PG_CHECKCALL
    end
    if Int(action) > Int(PG_CHECKCALL) && b.raises >= MAX_RAISES
        action = PG_CHECKCALL
    end
    a = convert_pg_action(b.big_blind, action)
    while a.amount + b.to_call > stack && Int(action) > Int(PG_CHECKCALL)
        action = PG_ACTION(Int(action) - 1)
        a=convert_pg_action(b.big_blind, action)
    end
    actionint = Int(action)
    probs[actionint] -= 1.0
    push!(strategy.experiences, PG3Experience(inputs, 0.0, action))
    return a
end