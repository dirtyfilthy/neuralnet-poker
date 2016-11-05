@enum ROUND prestart=0 preflop=1 flop=2 turn=3 river=4 finish=5

MAX_RAISES = 3

type Hand
    game::Game
    round::ROUND
    deck::Deck
    community::Array{Card, 1}
    pot::Int
    bet_round::Int
    hole_cards::Array{Array{Card,1},1}
    player_round_bets::Array{Int, 1}
    contributed::Array{Int, 1}
    active::Array{Boolean, 1}
    folded::Array{Boolean, 1}
    has_acted::Array{Boolean, 1}
    total_to_call::Int  
    player_turn::Int
    n_active::Int
    raises::Int

end

function Hand(game::Game)
    deck = Deck()
    hole_cards = Array{Array{Card,1},1}()
    active = Array{Boolean, 1}()
    has_acted = fill(false, game.n_players)
    folded = Array{Boolean, 1}()
    contributed = Array{Int, 1}()
    player_round_bets = Array{Int, 1}()
   
    for i in 1:game.n_players
       
        push!(hole_cards, sort(deal!(deck, 2), by=x->convert(Int64, x), rev=true))
        push!(player_round_bets, 0)
        push!(active, true)
        push!(folded, false)
        push!(contributed, 0)
    end
    community = Array{Card,1}()
    hand=Hand(game, prestart, deck, community, 0, 0, hole_cards, player_round_bets, contributed, active, folded, has_acted,
        0, 0, game.n_players, 0)
    next_round(hand)
    return hand
end

hand_to_call(hand::Hand, player_pos::Int) = hand.total_to_call - hand.player_round_bets[player_pos]

function hand_call(hand::Hand, blind::Boolean=false)
    game = hand.game
    player_pos = hand.player_turn
    to_call = hand_to_call(hand, player_pos)
    stack = game.players[player_pos].stack
    if(stack < to_call)
        put_in_chips(hand, player_pos, stack)
        make_inactive(hand, player_pos)
    else
        put_in_chips(hand, player_pos, to_call)
    end
    if !blind
        hand.has_acted[player_pos] = true
    end
    next_turn(hand)
    return nothing
end


function put_in_chips(hand::Hand, player_pos::Int, amount::Int)
    hand.player_round_bets[player_pos] += amount
    hand.bet_round += amount
    hand.pot += amount
    hand.contributed[player_pos] += amount
    if hand.player_round_bets[player_pos] > hand.total_to_call
        hand.total_to_call = hand.player_round_bets[player_pos]
    end
    hand.game.players[player_pos].stack -= amount
    return nothing

end

function hand_raise(hand::Hand, amount::Int, blind::Boolean=false)
    game = hand.game
    player_pos = hand.player_turn
    player_round_bets = hand.player_round_bets
    to_call = hand_to_call(hand, player_pos)
    total = amount + hand.total_to_call
    stack = game.players[player_pos].stack
    if amount == 0 
        return hand_call(hand)
    end
    if stack < to_call
        return hand_call(hand)
    end

    if hand.raises >= MAX_RAISES
        return hand_call(hand)
    end


    put_on_table = total - player_round_bets[player_pos]

    if stack < put_on_table
        put_on_table = stack 
    end
   
    put_in_chips(hand, player_pos, put_on_table)

    if !blind
        hand.has_acted[player_pos] = true
        
    end
    hand.raises += 1

    next_turn(hand)
    return nothing
end

function make_inactive(hand::Hand, player_pos::Int)
    if hand.active[player_pos]
        hand.active[player_pos] = false
        hand.n_active -= 1
    end
end

function hand_fold(hand::Hand)
    pos = hand.player_turn
    make_inactive(hand, pos)
    hand.folded[pos] = true
    hand.has_acted[pos] = true
    next_turn(hand)
    return nothing
end

function is_turn_finished(hand::Hand)
    if hand.n_active < 2
        return true
    end
    for i in 1:hand.game.n_players
        if hand.active[i] && (hand_to_call(hand, i) > 0 || !hand.has_acted[i])
            return false
        end
    end
    return true
end

function next_turn(hand::Hand)
    while true
        if hand.n_active < 2
            return nothing
        end
        hand.player_turn += 1
        if hand.player_turn > hand.game.n_players
            hand.player_turn = 1
        end
        hand.active[hand.player_turn] && break
    end
end



function next_round(hand::Hand)
    game = hand.game
    for i in 1:game.n_players
        hand.player_round_bets[i]=0
        hand.has_acted[i] = false
        if game.players[i].stack == 0 && hand.active[i]
            make_inactive(hand, i)
        end

    end
    hand.bet_round = 0
    hand.total_to_call = 0
    hand.raises = 0
    a = Int(hand.round)
    hand.round = ROUND(Int(hand.round) + 1)
    #println("before $a after $(Int(hand.round))")
    hand.player_turn = game.button
    next_turn(hand)

    if hand.round == flop
        hand.community = vcat(hand.community, deal!(hand.deck, 3))
    elseif hand.round == turn || hand.round == river
        hand.community = vcat(hand.community, deal!(hand.deck, 1))
    elseif hand.round == preflop

        hand_raise(hand, game.small_blind, true)
        hand_raise(hand, game.big_blind - game.small_blind, true)
        
    
    elseif hand.round == finish
        bad_score = HandScore(-999999, highcard, Array{Card, 1}())
        contributed = copy(hand.contributed)
        min_bet = 999999999999
        pot_left = hand.pot
        scores = Array{HandScore, 1}()
        all_scores = Array{HandScore, 1}()

        for i in 1:game.n_players
            score = evaluate(hand.community, hand.hole_cards[i])
            push!(all_scores, score)
            if hand.folded[i]
                push!(scores, bad_score)
            else
                push!(scores, score)
            end
        end
        max_overall = maximum(all_scores)
        would_have_won = map(x->x == max_overall, all_scores)
        #println("pot left $pot_left")
        while pot_left > 0
            active_by_position = Array{Int, 1}()
            sp = 0
            for i in 1:game.n_players
                if !hand.folded[i] && contributed[i] > 0
                    push!(active_by_position, i)
                    if contributed[i] < min_bet
                        min_bet = contributed[i]
                    end
                end
            end
            for i in 1:game.n_players
                c = min(contributed[i], min_bet)
                contributed[i] -= c
                sp += c
            end
            max_score = bad_score
            for i in active_by_position
                if scores[i].score > max_score.score
                    max_score = scores[i]
                end
            end
            #print("max score $max_score")
            winners = Array{Int, 1}()
            for i in active_by_position
                if scores[i].score == max_score.score
                    push!(winners, i)
                end
            end
            p = floor(sp / length(winners))
            #println("payout $p")
            payout = Int(p)
            for i in winners
                pot_left -= payout
                hand.game.players[i].stack += payout
                sp -= payout
                #println("Player $i wins $payout with $(string(scores[i].cards))")
            end
            if sp > 0
                pot_left -= sp
            end
        end
        return would_have_won
    end
end




