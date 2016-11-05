module Simulation

include("aliases.jl")
include("misc.jl")
include("card.jl")
include("deck.jl")
include("strategy.jl")
include("evaluator.jl")
include("player.jl")
include("game.jl")
include("hand.jl")


include("board.jl")
include("LayerParameter.jl")
include("NNLayer.jl")
include("GenericLayer.jl")
include("NN.jl")
include("experience.jl")
include("ExperienceStore.jl")
include("QLStrategy.jl")
include("QLStrategy2.jl")
include("PGStrategy.jl")
include("PGStrategy2.jl")
include("PGStrategy3.jl")
include("PGStrategy4.jl")
include("ActorCriticStrategy.jl")
export run_simulation
export NN
export add_layer
export feedforward
export backpropagate
export LinearLayer
export TanhLayer

PRINT_EVERY = 250


function heads_up()
    n_players = 2
    nn1 = construct_PG3()
    nn2 = construct_PG4()
    strat1 = PGStrategy3(nn1)
    strat2 = PGStrategy4(nn1)
    totals = [0, 0]
    for i in 1:1000000000
        g = Game()
        add_player(g, strat1)
        add_player(g, strat2)
        new_game(g)
        for h in 1:20
            totals += play_hand(g)
        end
        game_finished(g)
        if (i % 1000) == 0
            learn_from_experience!()
        end
        println("results at $i = $totals")
    end
end


function three_way()
    n_players = 2
    nn1 = construct_PG4()
    nn2 = construct_QL()
    nn3 = construct_QL2()
    strat1 = PGStrategy4(nn1)
    strat2 = QLStrategy(nn2)
    strat3 = QLStrategy2(nn3)
    totals = [0, 0, 0]
    for i in 1:10000000
        g = Game()
        add_player(g, strat1)
        add_player(g, strat2)
        add_player(g, strat3)
        new_game(g)
        for h in 1:20
            totals += play_hand(g)
        end
        game_finished(g)
        println("results at $i = $totals")
        if (i % 1000) == 0
            learn_from_experience!()
        end
    end
end

function four_way()
    n_players = 2
    nn1 = construct_QL()
    nn2 = construct_QL2()
    nn3 = construct_PG3()
    nn4 = construct_PG4()
    strat1 = ActorCriticStrategy()
    strat2 = QLStrategy2(nn2)
    strat3 = PGStrategy3(nn3)
    strat4 = PGStrategy4(nn3)
    totals = [0.0, 0.0, 0.0, 0.0]
    i2 = 0
    values = Vector{Vector{Int64}}()
    N = 2500
    hands_per_game = 25
    avg = fill(0.0, 4)
    for i in 1:100000000
        i2 += 1
        g = Game()
        add_player(g, strat1)
        add_player(g, strat2)
        add_player(g, strat3)
        add_player(g, strat4)
        new_game(g)
        temp = fill(0, 4)
        for h in 1:hands_per_game
            r = play_hand(g)
            temp += r
        end
        push!(values, temp)
        totals += temp
        if(length(values)>N)
            last = shift!(values)
            totals -= last
        end
        avg = (totals / hands_per_game) / length(values)
        game_finished(g)
        @printf("avg win/loss %7d %+6.2f %+6.2f %+6.2f %+6.2f\n", i, avg[1], avg[2], avg[3], avg[4])
        if (i % 1000) == 0
            learn_from_experience!()
        end
    end
end


function five_way()
    n_players = 2
    nn1 = construct_PG()
    nn2 = construct_PG2()
    nn3 = construct_PG3()
    nn4 = construct_QL()
    nn5 = construct_QL2()
    strat1 = PGStrategy(nn1)
    strat2 = PGStrategy2(nn2)
    strat3 = PGStrategy3(nn3)
    strat4 = QLStrategy(nn4)
    strat5 = QLStrategy2(nn5)
    totals = [0, 0, 0, 0, 0]
    for i in 1:100000
        g = Game()
        add_player(g, strat1)
        add_player(g, strat2)
        add_player(g, strat3)
        add_player(g, strat4)
        add_player(g, strat5)
        new_game(g)
        hands = 30 + intRand(30)
        for h in 1:hands
            totals += play_hand(g)
        end
        game_finished(g)
        println("results at $i = $totals")
    end
end

function run_simulation()
    t = fill(0.0, 9)
    it = 0

    nets = Array{NN, 1}(9)
    strategies = Array{Strategy, 1}(9)
    for i in 1:9
        nets[i] = construct_PG()
        strategies[i] = PGStrategy(nets[i])
    end

    

 

    for i in 1:20000
        

        g = Game()
        n_players = 1 + Int(floor(rand()*9))
        


        for j in 1:n_players
            add_player(g, strategies[j])
        end
        new_game(g)

        for h in 1:20
            res = play_hand(g)
            while length(res) < 9
                push!(res, 0.0)
            end
            t += res
            if mod(it, PRINT_EVERY) == 0
                println("game $(i) $(t / it)")
                t=fill(0.0, 9)
                #for layer in nn.layers
                #    println("biases $(layer.biases)")
                #end
            end
            it += 1
        end
        game_finished(g)
    end
    save(nets[1], "./policygradient.strategy")
end

end


