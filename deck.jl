import Base.shuffle!
type Deck
    remaining::Array{Card,1}
    
    function Deck()
        d = new(Array{Card,1}())
        reset!(d)
        shuffle!(d)
        return d
    end

end

function reset!(d::Deck)
    d.remaining = Array{Card, 1}()
    for suit in Int(diamonds):Int(clubs)
        for rank in Int(two):Int(ace)
            push!(d.remaining, Card(SUIT(suit), RANK(rank)))
        end
    end
end

shuffle!(d::Deck) = shuffle!(d.remaining)

deal!(d::Deck) = pop!(d.remaining)

function deal!(d::Deck, n::Int) 
    a = Vector{Card}()
    for i in 1:n
        push!(a, deal!(d))
    end
    return a
end



