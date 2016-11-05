typealias PlayerRoundHistory Vector{Vector{Action}}

type Game
    players::Array{Player, 1}
    button::Int
    starting_stack::Int
    n_players::Int
    big_blind::Int
    small_blind::Int
    action_history::Vector{Vector{Vector{Action}}}
end

function Game()
    return Game(Array{Player,1}(), 0, 1000, 0, 20, 10, Vector{PlayerRoundHistory}())
end

function new_game(game::Game)
    game.button = intRandFromOne(game.n_players)
    for p in game.players
        new_game(p.strategy)
    end

end

function new_hand(g::Game)
    g.action_history = Vector{PlayerRoundHistory}()
    for round in 1:4
        push!(g.action_history, PlayerRoundHistory())
    end
    for player in g.players
        for round in 1:4
            push!(g.action_history[round], Vector{Action}())
        end
        new_hand(player.strategy)
    end

end


function game_finished(g::Game)
    for p in g.players
        game_finished(p.strategy)
    end
end

function add_player(game::Game, strategy::Strategy)
    p = Player(game.starting_stack, strategy)
    game.n_players += 1
    push!(game.players, p)
    
end

function play_hand(game::Game)
    starting_stacks = map(x->x.stack, game.players)
    hand = Hand(game)
    new_hand(game)
    would_have_won = nothing
    while hand.round != finish
        #println(hand.round)
        while(!is_turn_finished(hand))
            play_turn(game, hand)
        end
        would_have_won = next_round(hand)
    end
    sc = -99999
    results = map(x->x.stack, game.players) - starting_stacks
    growth = map(x->Float(x.stack), game.players) ./ starting_stacks
    player_update = Int(floor((rand() * (game.n_players-1) + 2)))
    game.button += 1
    if game.button > game.n_players
        game.button = 1
    end
    for i in 1:game.n_players
        sc = max(sc, hand_finished(game.players[i].strategy, results[i], growth[i], would_have_won[i], starting_stacks[i]))
    end  
    return results
end

function play_turn(game::Game, hand::Any)
    current_player = hand.player_turn
    hole_cards = hand.hole_cards[current_player]
    stacks = calc_rel_array(map(x->x.stack, game.players), current_player)
    active = calc_rel_array(hand.active, current_player)
    folded = calc_rel_array(hand.folded, current_player)
    to_call = hand_to_call(hand, current_player)
    hand_history = Vector{PlayerRoundHistory}(4)
    for round in 1:4
        hand_history[round] = calc_rel_array(game.action_history[round], current_player)
    end
    contributed = hand.contributed[current_player]
    pot = hand.pot
    r = Int(hand.round)
    board = Board(hand.round, stacks, active, folded, hole_cards, hand.community, hand.pot, to_call, 
        calc_rel_pos(game, game.button, current_player), game.big_blind, game.starting_stack, 
        contributed, hand.raises, hand_history)
    action = get_action(game.players[current_player].strategy, board)
    push!(game.action_history[r][current_player], action)
    # println("action $(action.action) $(action.amount)")
    if action.action == fold
        hand_fold(hand)
    elseif action.action == call
        hand_call(hand)
    elseif action.action == raise
        hand_raise(hand, action.amount)
    end
end


calc_rel_pos(game::Game, player_pos::Int, pos::Int) = mod(pos - player_pos, game.n_players) + 1

function calc_rel_array{T<:Array}(ary::T, player_pos::Int) 
    per = copy(ary)
    i = player_pos
    while i > 1
        push!(per, shift!(per))
        i = i - 1
    end
    return per
end
