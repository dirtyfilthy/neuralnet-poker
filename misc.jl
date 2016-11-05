const RNG = RandomDevice()
const FIND_EVIL_NAN = false
randfloat(args...) = rand(RNG, Float, args...)
fillfloat(n, args...) = fill(Float(n), args...)

function save(obj::Any, path::String)
    open(path, "w") do f
        serialize(f, obj)
    end
end

function load(path::String)
    open(path, "r") do f
        return deserialize(f)
    end
end

function pickOne{T}(a::Vector{T})
    n = length(a)
    i = intRandFromOne(n)
    return a[i]
end

function heading(h::String)
    n = length(h)
    println()
    println("=" ^ n)
    println(h)
    println("=" ^ n)
end

function safeLog(x::Float)
    if isnan(x) || x <= 0.0 
        return log(0.00000000001)
    end
    return log(x)
end



function findEvilNan{T<:AbstractFloat}(message::String, v::Vector{T})
    if FIND_EVIL_NAN && findfirst(x->isnan(x), v) !=0
        println("FOUND NAN ($message) in $v")
        exit()
    end
end

function findEvilNan{T<:AbstractFloat}(message::String, v::Matrix{T})
    if FIND_EVIL_NAN && findfirst(x->isnan(x), v) !=0
        println("FOUND NAN ($message) in $v")
        exit()
    end
end



safeLog(x::Int) = safeLog(Float(x))

safeLog(x::OtherFloat) = safeLog(Float(x))


intRand(x::Int) = Int(floor(rand() * x))
intRandFromOne(x::Int) = intRand(x) + 1

profitToFloat(big_blind::Int, profit::Int) = Float((profit / big_blind) / 100.0)

floatToProfit(big_blind::Int, f::Float) = Float(f * big_blind * 100.0)