
PG4_OUTPUT_LAYER = 8
PG4_HIDDEN_LAYER = 120
PG4_HIDDEN_LAYER2 = 80
PG4_HIDDEN_LAYER3 = 60
PG4_HIDDEN_LAYER4 = 40
PG4_HIDDEN_LAYER5 = 20
PG4_EXPLOIT = 0.05

type PGStrategy4 <: Strategy
    nn::NN
    experiences::Vector{Experience}
    baseline::Float
end

function construct_PG4()
    nn = NN()
    nn.batch_size = 250
    add_layer(nn, PReLULayer(N_INPUT_LAYER, PG4_HIDDEN_LAYER))
    add_layer(nn, PReLULayer(PG4_HIDDEN_LAYER, PG4_HIDDEN_LAYER2))
    add_layer(nn, PReLULayer(PG4_HIDDEN_LAYER2, PG4_HIDDEN_LAYER3))
    add_layer(nn, PReLULayer(PG4_HIDDEN_LAYER3, PG4_HIDDEN_LAYER4))
    add_layer(nn, PReLULayer(PG4_HIDDEN_LAYER4, PG4_HIDDEN_LAYER5))
    add_layer(nn, SoftMaxLayer(PG4_HIDDEN_LAYER5, PG4_OUTPUT_LAYER))
    return nn
end

type PG4Experience <: Experience 
    inputs::InputVector
    reward::Float64
    action::PG_ACTION
    stack::Int64
    stack_after::Int64
    would_have_won::Bool
end

PG4Experience(inputs::InputVector, action::PG_ACTION, stack::Int) = PG4Experience(inputs, 0.0, action, stack, 0.0, false)

function deltas(nn::NN, e::PG4Experience, outputs::OutputVector)
    targets = copy(outputs)
    targets[Int(e.action)] -= 1.0
    return targets * e.reward
end



PGStrategy4(nn::NN) = PGStrategy4(nn, Vector{Experience}(), 0.0)




function new_game(strategy::PGStrategy4)

end



function new_hand(strategy::PGStrategy4)
    strategy.experiences = Vector{PG3Experience}()

end


function is_log_space(strategy::PGStrategy4)
    return true
end

function hand_finished(strategy::PGStrategy4, result::Int, growth::Float, would_have_won::Bool, stack_before::Int)
    stack_after = stack_before + result

    for experience in strategy.experiences
        experience.reward = safeLog(growth) - strategy.baseline
        experience.stack_after = stack_after
        experience.would_have_won = would_have_won
    end
    add_experience!(strategy, strategy.experiences)
    strategy.baseline = (0.9 * strategy.baseline) + (0.1 * safeLog(growth))
    return result

end

function game_finished(strategy::PGStrategy4)
 

end



function get_action(strategy::PGStrategy4, b::Board)
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
    if rand() < PG4_EXPLOIT
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
    push!(strategy.experiences, PG4Experience(inputs, action, stack))
    return a
end