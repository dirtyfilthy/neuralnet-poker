using Combinatorics
@enum HAND_CATEGORY highcard=0 pair=1 twopair=2 threeofakind=3 straight=4 flush=5 fullhouse=6 fourofakind=7 straightflush=8
import Base.==
import Base.isless
type HandScore
    score::Int
    category::HAND_CATEGORY
    cards::Array{Card, 1}
end



function isless(a::HandScore, b::HandScore)
    return a.score < b.score
end

function ==(a::HandScore, b::HandScore)
    return a.score == b.score
end

function score(category::HAND_CATEGORY, s::Array{RANK})
    score = Int(category)
    while length(s) < 5
        push!(s, bad_rank)
    end
    for i in 1:5
        score = score << 4
        score = score | Int(s[i])
    end
    return score, category
end

function is_flush(hand::Array{Card})
    suit = hand[1].suit
    for i in 2:5
        if hand[i].suit != suit
            return false
        end
    end
    return true
end

function is_straight(hand::Array{Card})
    last_rank = Int(hand[1].rank)
    for i in 2:5
        rank = Int(hand[i].rank)
        if i == 5 && last_rank == 5 && rank == Int(ace)
            return true, five 
        end
        if rank != last_rank + 1
            return false, bad_rank
        end
        last_rank = rank
    end
    return true, hand[5].rank
end


function evaluate(community::Array{Card}, holecards::Array{Card})
    allcards = vcat(community, holecards)
    topscore = 0
    tophand = nothing
    toptype = nothing
    for hand in combinations(allcards, 5)
        s, t = evaluate(hand)
        if s > topscore
            topscore = s
            tophand = hand
            toptype = t
        end
    end
    # tophand = sort!(tophand, by=x->convert(Int64, x), rev=true)
    return HandScore(topscore, toptype, tophand)
end



function evaluate(hand::Array{Card})
    sorted_hand = sort(hand, by=x->Int(x.rank))
    maybe_straight, max_rank = is_straight(sorted_hand)
    maybe_flush = is_flush(sorted_hand)
    rank_map = reverse(map(x->x.rank, sorted_hand))
    if maybe_straight && maybe_flush
        return score(straightflush, [max_rank])
    end
    if maybe_flush 
        return score(flush, rank_map)
    end
    if maybe_straight 
        return score(straight, [max_rank])
    end
    histogram = Dict{RANK, Int}()
    threes = Array{RANK,1}()
    twos = Array{RANK,1}()
    ones = Array{RANK,1}()
    for i in 1:5
        rank = sorted_hand[i].rank
        if !haskey(histogram, rank)
            histogram[rank] = 1
        else
            histogram[rank] += 1
            if histogram[rank] == 4
                return score(fourofakind,[rank])
            end
        end
       
    end
    for i in unique(rank_map)
        if histogram[i] == 2
            push!(twos, i)
        end
        if histogram[i] == 3
            push!(threes, i)
        end
         if histogram[i] == 1
            push!(ones, i)
        end
    end
    sort!(ones, by=x->Int(x), rev=true)
    sort!(twos, by=x->Int(x), rev=true)
    if length(threes) > 0
        if length(twos) > 0
            return score(fullhouse, vcat(threes, twos))
        else
            return score(threeofakind, vcat(threes, ones))
        end
    end
    if length(twos) == 2
        return score(twopair, vcat(twos, ones))
    end
    if length(twos) == 1
        return score(pair, vcat(twos, ones))
    end
    return score(highcard, ones)

end