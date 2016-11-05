type Board
    round::ROUND
    stacks::Array{Int, 1}
    active::Array{Boolean, 1}
    folded::Array{Boolean, 1}
    hole_cards::Array{Card, 1}
    community::Array{Card, 1}
    pot::Int
    to_call::Int
    position::Int
    big_blind::Int
    starting_stack::Int
    contributed::Int
    raises::Int
    action_history::Vector{Vector{Vector{Action}}}
end

MAX_PLAYERS = 9
N_HOLE_CARDS= 2
N_COMMUNITY_CARDS= 5
N_TOTAL_CARDS = N_HOLE_CARDS + N_COMMUNITY_CARDS
N_SINGLE_CARD_ENCODING= 17
N_ROUND_ENCODING = 4
N_POT = 1
N_TO_CALL = 1
N_CONTRIBUTED = 1
N_POT_ODDS = 1
N_TO_CALL_PROP = 1
N_RAISE_BB_PROP = 1
N_RAISE_2BB_PROP = 1
N_RAISE_4BB_PROP = 1
N_RAISE_8BB_PROP = 1
N_RAISE_16BB_PROP = 1
N_RAISE_32BB_PROP = 1
N_CONTRIBUTED_PROP = 1
N_FOLD_PROP = 1
N_RAISES_ENCODING = MAX_RAISES + 1
N_PROP = N_TO_CALL_PROP + N_RAISE_BB_PROP + N_RAISE_2BB_PROP + N_RAISE_4BB_PROP + N_RAISE_8BB_PROP + N_RAISE_16BB_PROP + 
         N_RAISE_32BB_PROP  + N_CONTRIBUTED_PROP + N_FOLD_PROP
N_STACKS = 1
N_ACTIVE = 1
N_FOLDED = 1
N_POSITION = MAX_PLAYERS
N_PLAYER_NO = MAX_PLAYERS
N_RAISED_LAST_ROUND = 1
N_SURVIVABLE = 1
N_BOOLS = (N_STACKS + N_ACTIVE + N_FOLDED + N_RAISED_LAST_ROUND) * MAX_PLAYERS
N_MISC = N_POT + N_TO_CALL + N_CONTRIBUTED + N_POT_ODDS
N_HOLE_CARD_PROPERTIES = 3
N_COMMUNITY_CARD_PROPERTIES = (10 * 2)
N_TOTAL_CARD_ENCODING = N_SINGLE_CARD_ENCODING * N_TOTAL_CARDS + N_RAISES_ENCODING
N_INPUT_LAYER = N_ROUND_ENCODING + N_BOOLS + N_TOTAL_CARD_ENCODING + N_MISC + 
                N_HOLE_CARD_PROPERTIES + N_COMMUNITY_CARD_PROPERTIES  + N_PLAYER_NO + N_POSITION + N_PROP

function encodeCountNeurons(x::Int, max::Int)
    output = fillfloat(0.0, max)
    output[x] = 1.0
    return output
end

function encodeInputNeurons(b::Array{Boolean,1}, size::Int)
    o = fillfloat(0.0, size)
    for i in 1:length(b)
        if b[i]
            o[i] = 1.0
        end
    end
    return o
end

encodeInputNeurons(x::Float) = [x]
encodeInputNeurons(x::Float, max::Float) = [hard_max(x, max)]
encodeInputNeurons(x::Float, max::Int) = [hard_max(x, Float(max))]
encodeLogInputNeurons(x::Float, max) = [x==0 ? log(0.0000000001) : log(hard_max(x, max))]
encodeLogInputNeurons(x::Float, max::Int) = encodeLogInputNeurons(x, Float(max))
encodeLogInputNeurons(x::Float) = [x==0 ? log(0.0000000001) : log(x)]
encodeMaybeLog(is_log::Bool, x::Float) = is_log ? encodeLogInputNeurons(x) : encodeInputNeurons(x)
encodeMaybeLog(is_log::Bool, x::Float, max::Float) = is_log ? encodeLogInputNeurons(x, max) : encodeInputNeurons(x, max)
encodeMaybeLog(is_log::Bool, x::Float, max::Int) = is_log ? encodeLogInputNeurons(x, Float(max)) : encodeInputNeurons(x, Float(max))
encodeMaybeLog(is_log::Bool, x::Int, max::Int) = is_log ? encodeLogInputNeurons(Float(x), Float(max)) : encodeInputNeurons(Float(x), Float(max))
encodeMaybeLog(is_log::Bool, x::OtherFloat) = encodeMaybeLog(is_log, Float(x))
encodeMaybeLog(is_log::Bool, x::OtherFloat, max::Float) = encodeMaybeLog(is_log, Float(x), max)
encodeMaybeLog(is_log::Bool, x::OtherFloat, max::OtherFloat) = encodeMaybeLog(is_log, Float(x), Float(max))

function hard_max(x::Float, max::Float)
    v = x > max ? max : x
    if max == 0.0
        return 0.0
    end
    return v / max
end

hard_max(x::OtherFloat, max::OtherFloat) = hard_max(Float(x), Float(max))

function encode_stacks(b::Board, size::Int, s::Strategy)
    n_players = length(b.stacks)
    output = fillfloat(0.0, size)
    m = (b.starting_stack / b.big_blind) * 10.0
    if is_log_space(s)
        for i in 1:n_players
            x = b.stacks[i] / b.big_blind
            output[i] = x==0 ? log(0.0000000001) : log(x)
        end
    else
        m = (b.starting_stack / b.big_blind) * 10.0
        for i in 1:n_players
            x = b.stacks[i] / b.big_blind
            output[i] = hard_max(x, m)
        end
    end

    return output
end

function encodeRaisedLastRound(b::Board)
    if b.round == preflop
        return fillfloat(0.0, MAX_PLAYERS)
    end
    n_players = length(b.stacks)
    ret = fillfloat(0.0, MAX_PLAYERS)
    last_round = Int(b.round) - 1
    for p in 1:n_players
        actions = b.action_history[last_round][p]
        if findfirst(x->x.action == raise, actions) != 0
            ret[p] = 1.0
        end
    end
    return ret

end



function encode_board(b::Board, strategy::Strategy)
    n_players = length(b.stacks)
    pot_odds = b.to_call / (b.to_call + b.pot)
    stack = b.stacks[1]
    if stack == 0
        stack = 1
    end

    input_neurons = Vector{Float}()
    sizehint!(input_neurons, N_INPUT_LAYER)
    append!(input_neurons, encodeCountNeurons(Int(b.round), 4))
    append!(input_neurons, encode_stacks(b, MAX_PLAYERS, strategy))
    append!(input_neurons, encodeInputNeurons(b.active, MAX_PLAYERS))
    append!(input_neurons, encodeInputNeurons(b.folded, MAX_PLAYERS))
    append!(input_neurons, encodeInputNeurons(b.hole_cards, 2))
    append!(input_neurons, encodeInputNeurons(b.community, 5))
    append!(input_neurons, encodeCountNeurons(n_players, MAX_PLAYERS))
    append!(input_neurons, encodeCountNeurons(b.position, MAX_PLAYERS))
    append!(input_neurons, encodeCountNeurons(b.raises+1, MAX_RAISES+1))
    append!(input_neurons, encodeRaisedLastRound(b))


    is_log = is_log_space(strategy)
   
    append!(input_neurons, encodeMaybeLog(is_log, b.pot, stack))
    append!(input_neurons, encodeMaybeLog(is_log, b.to_call / b.big_blind))
    append!(input_neurons, encodeMaybeLog(is_log, b.contributed / b.big_blind))
    append!(input_neurons, encodeMaybeLog(is_log, pot_odds)) 
    append!(input_neurons, encodeMaybeLog(is_log, b.to_call, stack))  
    append!(input_neurons, encodeMaybeLog(is_log, b.big_blind + b.to_call, stack))
    append!(input_neurons, encodeMaybeLog(is_log, b.big_blind*2 + b.to_call, stack))
    append!(input_neurons, encodeMaybeLog(is_log, b.big_blind*4 + b.to_call, stack))
    append!(input_neurons, encodeMaybeLog(is_log, b.big_blind*8 + b.to_call, stack))
    append!(input_neurons, encodeMaybeLog(is_log, b.big_blind*16 + b.to_call, stack))
    append!(input_neurons, encodeMaybeLog(is_log, b.big_blind*32 + b.to_call, stack))
    append!(input_neurons, encodeMaybeLog(is_log, b.contributed, stack + b.contributed))
    append!(input_neurons, encodeMaybeLog(is_log, stack, stack + b.contributed))

    for c in b.hole_cards
        pairs = map(x->x.rank == c.rank, b.community)

        append!(input_neurons, encodeInputNeurons(pairs, 5)) 
        suited = map(x->x.suit == c.suit, b.community)
        append!(input_neurons, encodeInputNeurons(suited, 5))
    end 
    hole_prop = [0.0, 0.0, 0.0]
    if b.hole_cards[1].rank == b.hole_cards[2].rank
        hole_prop[1] = 1.0
    end
    if b.hole_cards[1].suit == b.hole_cards[2].suit
        hole_prop[2] = 1.0
    end
    if abs(Int(b.hole_cards[1].rank) - Int(b.hole_cards[2].rank)) == 1
        hole_prop[3] = 1.0
    end
    append!(input_neurons, hole_prop) 
    return input_neurons
end

