@enum AC_ACTION AC_FOLD=1 AC_CHECKCALL=2 AC_RAISEBB=3 AC_RAISE2BB=4 AC_RAISE4BB=5 AC_RAISE8BB=6 AC_RAISE16BB=7 AC_RAISE32BB=8

typealias PolicyVector Vector{Float}
HIGHER_REWARD_DELTA = 10.0


AC_N_ACTIONS = 8
AC_EXPLOIT = 0.01



CRITIC_INPUT_LAYER = N_INPUT_LAYER + AC_N_ACTIONS
CRITIC_HIDDEN_LAYER1 = 120
CRITIC_HIDDEN_LAYER2 = 80
CRITIC_HIDDEN_LAYER3 = 60
CRITIC_HIDDEN_LAYER4 = 40
CRITIC_HIDDEN_LAYER5 = 20
CRITIC_OUTPUT_LAYER = 1

ACTOR_INPUT_LAYER = N_INPUT_LAYER
ACTOR_HIDDEN_LAYER1 = 120
ACTOR_HIDDEN_LAYER2 = 80
ACTOR_HIDDEN_LAYER3 = 60
ACTOR_HIDDEN_LAYER4 = 40
ACTOR_HIDDEN_LAYER5 = 20
ACTOR_OUTPUT_LAYER = AC_N_ACTIONS




PG4_EXPLOIT = 0.05

type ActorCriticStrategy <: Strategy
    actor::NN
    critic::NN
    experiences::Vector{Experience}
end

type ActorCriticExperience <: Experience 
    board_inputs::InputVector
    policy::PolicyVector
    action::AC_ACTION
    reward::Float
end

ActorCriticExperience(board_inputs::InputVector, policy::PolicyVector, action::AC_ACTION) = ActorCriticExperience(board_inputs, policy, action, 0.0)


function ActorCriticStrategy()
    actor = construct_actor()
    critic = construct_critic()
    experiences = Vector{Experience}()
    ac = ActorCriticStrategy(actor, critic, experiences)
    return ac
end

function construct_critic()
    nn = NN()
    add_layer(nn, PReLULayer(CRITIC_INPUT_LAYER,   CRITIC_HIDDEN_LAYER1))
    add_layer(nn, PReLULayer(CRITIC_HIDDEN_LAYER1, CRITIC_HIDDEN_LAYER2))
    add_layer(nn, PReLULayer(CRITIC_HIDDEN_LAYER2, CRITIC_HIDDEN_LAYER3))
    add_layer(nn, PReLULayer(CRITIC_HIDDEN_LAYER3, CRITIC_HIDDEN_LAYER4))
    add_layer(nn, PReLULayer(CRITIC_HIDDEN_LAYER4, CRITIC_HIDDEN_LAYER5))
    add_layer(nn, PReLULayer(CRITIC_HIDDEN_LAYER5, CRITIC_OUTPUT_LAYER))
    return nn
end

function construct_actor()
    nn = NN()
    add_layer(nn, PReLULayer(   ACTOR_INPUT_LAYER,   ACTOR_HIDDEN_LAYER1))
    add_layer(nn, PReLULayer(   ACTOR_HIDDEN_LAYER1, ACTOR_HIDDEN_LAYER2))
    add_layer(nn, PReLULayer(   ACTOR_HIDDEN_LAYER2, ACTOR_HIDDEN_LAYER3))
    add_layer(nn, PReLULayer(   ACTOR_HIDDEN_LAYER3, ACTOR_HIDDEN_LAYER4))
    add_layer(nn, PReLULayer(   ACTOR_HIDDEN_LAYER4, ACTOR_HIDDEN_LAYER5))
    add_layer(nn, SoftMaxLayer( ACTOR_HIDDEN_LAYER5, ACTOR_OUTPUT_LAYER))
    return nn
end


function get_action(strategy::ActorCriticStrategy, b::Board)
      #action = QL_ACTION(7 - Base.convert(Int, floor(log2(1.0+rand()*127))))
    stack = b.stacks[1]
    fold_value = -b.contributed
    inputs = encode_board(b, strategy)
    outputs = feedforward(strategy.actor, inputs)
    policy = outputs[end]
    c = 0.0
    r = rand()
    action = AC_FOLD
    for i in 1:length(policy)
        c += policy[i]
        if r < c
            action = AC_ACTION(i)
            break
        end
    end

    if action == AC_FOLD && b.to_call == 0
        action = AC_CHECKCALL
    end
    if Int(action) > Int(AC_CHECKCALL) && b.raises >= MAX_RAISES
        action = AC_CHECKCALL
    end
    a = convert_ac_action(b.big_blind, action)
    while a.amount + b.to_call > stack && Int(action) > Int(AC_CHECKCALL)
        action = AC_ACTION(Int(action) - 1)
        a=convert_ac_action(b.big_blind, action)
    end
    actionint = Int(action)
    push!(strategy.experiences, ActorCriticExperience(inputs, policy, action))
    return a
end

function convert_ac_action(bb::Int, action::AC_ACTION)
    if action == AC_FOLD
        return Action(fold, 0)
    elseif action == AC_CHECKCALL
        return Action(call, 0)
    elseif action == AC_RAISEBB
        return Action(raise, bb)
    elseif action == AC_RAISE2BB
        return Action(raise, 2*bb)
    elseif action == AC_RAISE4BB
        return Action(raise, 4*bb)
    elseif action == AC_RAISE8BB
        return Action(raise, 8*bb)
    elseif action == AC_RAISE16BB
        return Action(raise, 16*bb)
    else action == AC_RAISE32BB
        return Action(raise, 32*bb)
    end
end

function critic_deltas(strategy::ActorCriticStrategy, board_inputs::InputVector, outputs::OutputList, policy::PolicyVector, reward::Float; direct_delta=false)
    inputs = vcat(board_inputs, policy)
    estimate_delta = 0.0
    if direct_delta
        estimate_delta = reward
    else
        estimate = outputs[end][1]
        estimate_delta = estimate - reward
    end
    full_deltas = calc_deltas(strategy.critic, outputs, [estimate_delta])
    policy_deltas = vec(strategy.critic.layers[1].weights * full_deltas[1])[(N_INPUT_LAYER+1):end]
    return full_deltas, policy_deltas
end

function critic_inputs(experience::ActorCriticExperience)
    return hcat(experience.board_inputs, experience.policy)
end

function critic_training_inputs(strategy::ActorCriticStrategy, experience_list::Vector{Experience}, batch_size::Int)
    batch_inputs = Matrix{Float}(CRITIC_INPUT_LAYER, batch_size*3)
    batch_experiences = Vector{Experience}()
    @simd for i in 1:batch_size
        experience = pickOne(experience_list)
        action_policy     = fillfloat(0.0, length(experience.policy))
        standard_policy   = experience.policy
        preturbed_policy  = map(x->x+x*randfloat(), standard_policy)
        preturbed_policy /= sum(preturbed_policy)
        @inbounds batch_inputs[:, 3(i - 1) + 1] = vcat(experience.board_inputs, action_policy)
        @inbounds batch_inputs[:, 3(i - 1) + 2] = vcat(experience.board_inputs, preturbed_policy)
        @inbounds batch_inputs[:, 3(i - 1) + 3] = vcat(experience.board_inputs, standard_policy)
        push!(batch_experiences, experience)
    end
    return batch_inputs, batch_experiences
end

function train_batch!(strategy::ActorCriticStrategy, store::ExperienceStore)
    batch_size = store.batch_size
    experience_list = all_experiences(strategy, store)
    batch_inputs, batch_experiences = critic_training_inputs(strategy, experience_list, batch_size)
    batch_outputs = feedforward(strategy.critic, batch_inputs)
    batch_targets = OutputMatrix(store.batch_size * 3, 1)
    idx = 1
    @simd for i in 1:batch_size
        @inbounds batch_targets[3(i - 1) + 1, 1] = batch_experiences[i].reward
        @inbounds batch_targets[3(i - 1) + 2, 1] = batch_experiences[i].reward
        @inbounds batch_targets[3(i - 1) + 3, 1] = batch_experiences[i].reward
    end

    batch_initial_deltas = batch_outputs[end] .- batch_targets
    backpropagate!(strategy.critic, batch_initial_deltas, batch_outputs)
    actor_delta_inputs = Matrix{Float}(CRITIC_INPUT_LAYER, batch_size)

    @simd for i in 1:batch_size
         @inbounds actor_delta_inputs[:, i] = batch_inputs[:, 3(i - 1) + 3]
    end

    actor_delta_outputs        = feedforward(strategy.critic, actor_delta_inputs)
    actor_delta_initial_deltas = fillfloat(-HIGHER_REWARD_DELTA, batch_size, 1)
    actor_delta_deltas         = calc_deltas(strategy.critic, actor_delta_outputs, actor_delta_initial_deltas)
    actor_policy_deltas        = (strategy.critic.layers[1].weights * actor_delta_deltas[1]')[(N_INPUT_LAYER+1):end, :]
    actor_policy_inputs        = Matrix{Float}(ACTOR_INPUT_LAYER, batch_size)

    @simd for i in 1:batch_size
        @inbounds actor_policy_inputs[:, i] = batch_experiences[i].board_inputs
    end

    actor_policy_outputs = feedforward(strategy.actor, actor_policy_inputs)
    backpropagate!(strategy.actor, actor_policy_deltas', actor_policy_outputs)

    update!(strategy.critic)
    update!(strategy.actor)

end


function backpropagate_critic!(strategy::ActorCriticStrategy, experience::Experience)
    
    full_deltas::OutputList = OutputList()
    critic_outputs::OutputList = OutputList()

    standard_policy = experience.policy
    action_policy = fillfloat(0.0, length(experience.policy))
    action_policy[Int(experience.action)] = 1.0
    preturbed_policy = map(x->x+x*randfloat(), standard_policy)
    preturbed_policy /= sum(preturbed_policy)
    actor_deltas = nothing
    policies = [action_policy, preturbed_policy, standard_policy]
    for policy in policies
        inputs = vcat(experience.board_inputs, policy)
        critic_outputs = feedforward(strategy.critic, inputs)
        full_deltas, actor_deltas = critic_deltas(strategy, experience.board_inputs, critic_outputs, policy, experience.reward)
        backpropagate!(strategy.critic, full_deltas, critic_outputs)
    end 
    predicted_reward = critic_outputs[end][1]
    full_deltas, actor_deltas = critic_deltas(strategy, experience.board_inputs, critic_outputs, standard_policy, Float(HIGHER_REWARD_DELTA), direct_delta=true)
    return actor_deltas
end

function backpropagate!(strategy::ActorCriticStrategy, experience::Experience)
    actor_deltas  = backpropagate_critic!(strategy, experience)
    actor_outputs = feedforward(strategy.actor, experience.board_inputs)
    backpropagate!(strategy.actor, actor_deltas, actor_outputs)
end

function train_strategy!(strategy::ActorCriticStrategy, store::ExperienceStore)
    println("[learning $(typeof(strategy))]")
    experience_list = all_experiences(strategy, store)
    for batch in 1:store.max_batches
        if (batch % 5) == 0
            println("Processing batch $batch")
        end
        train_batch!(strategy, store)
    end
end








function new_game(strategy::ActorCriticStrategy)

end



function new_hand(strategy::ActorCriticStrategy)
    strategy.experiences = Vector{ActorCriticExperience}()

end


function is_log_space(strategy::ActorCriticStrategy)
    return true
end

function hand_finished(strategy::ActorCriticStrategy, result::Int, growth::Float, would_have_won::Bool, stack_before::Int)

    for experience in strategy.experiences
        experience.reward = safeLog(growth)
    end
    add_experience!(strategy, strategy.experiences)
    return result

end

function game_finished(strategy::ActorCriticStrategy)
 

end


