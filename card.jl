import Base.string
@enum SUIT bad_suit=0 diamonds=1 hearts=2 spades=3 clubs=4
@enum RANK bad_rank=0 two=2 three=3 four=4 five=5 six=6 seven=7 eight=8 nine=9 ten=10 jack=11 queen=12 king=13 ace=14

const suit_to_string = Dict{SUIT, String}(diamonds => "d", hearts => "h", spades => "s", clubs => "c")
const rank_to_string = Dict{RANK, String}(  two => "2", 
                                            three => "3",
                                            four => "4",
                                            five => "5",
                                            six => "6",
                                            seven => "7",
                                            eight => "8",
                                            nine => "9",
                                            ten => "T",
                                            jack => "J",
                                            queen => "Q",
                                            king => "K",
                                            ace => "A")

const char_to_suit = Dict{Char, SUIT}('d'=>diamonds, 'h' => hearts, 's'=>spades, 'c'=>clubs )
const char_to_rank = Dict{Char, RANK}(  '2' => two, 
                                            '3' => three,
                                            '4' => four,
                                            '5' => five,
                                            '6' => six,
                                            '7' => seven,
                                            '8' => eight,
                                            '9' => nine,
                                            'T' => ten,
                                            'J' => jack,
                                            'Q' => queen,
                                            'K' => king,
                                            'A' => ace)


type Card
    suit::SUIT
    rank::RANK
end

macro cards_str(s)
    convert(Array{Card}, s)
end

function convert(::Type{String}, c::Card)
    return "$(rank_to_string[c.rank])$(suit_to_string[c.suit])"
end



function convert(::Type{Int64}, c::Card)
    return (Int(c.suit) - 1) + (4 * (Int(c.rank) - 2))
end

function convert(::Type{Array{Card}}, s::String)
    return map(x->convert(Card, x), split(s))
end

function convert(::Type{String}, a::Array{Card, 1})
    return join(map(x->convert(String, x), a), " ")
end

function encodeInputNeurons(cards::Array{Card, 1}, max_cards::Int)
    output = fill(0.0, max_cards * 17)
    offset = 0
    for card in cards
        output[Int(card.suit) + offset] = 1.0
        output[Int(card.rank) + 3 + offset] = 1.0
        offset += 17
    end
    return output
end




function convert(::Type{Card}, s::AbstractString)
    suit = bad_suit
    rank = bad_rank
    if length(s) < 2
        error("String is too short to convert to card")
    end
    r = uppercase(s[1])
    s = lowercase(s[2])
    if !haskey(char_to_suit, s)
        error("'$s' is not a valid suit")
    end
    suit = char_to_suit[s]
    if !haskey(char_to_rank, r)
        error("'$r' is not a valid rank")
    end
    rank = char_to_rank[r]
    return Card(suit, rank)
end

function string(c::Card)
    return convert(String, c)
end

function string(a::Array{Card})
    return convert(String, a)
end
