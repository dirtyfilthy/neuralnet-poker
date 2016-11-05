type Solution
    volume::Float64
    ingredient::Float64
end

+(a::Solution, b::Solution) = Solution(a.volume + b.volume, a.ingredient + b.ingredient)
+(a::Solution, b::Float64) = a + Solution(b, 0.0)
dilute(a::Solution, b::Float64) = a + b
undilute(a::Solution, b::Float64) = a + Solution(0.0, b)
*(a::Solution, b::Float64) = Solution(a.volume * b, a.ingredient * b)
strength(a::Solution) = a.ingredient / a.volume
create_from(a::Solution, amt::Float64) = Solution(amt, strength(a) * amt)




last = Solution(0.0, 0.0)
jars = 2.5
output = 0.8
ref = Solution(800, 3.6)
for i = 1:100
    to_add = 1000 - last.volume
    add = create_from(ref, to_add)
    println("adding $add to solution of $(last) + $jars jars")
    new_v = dilute(last, to_add)
    new_v = undilute(new_v, jars)
    println("output of $output")
    new_v = new_v * output
    println("solution is $new_v")
    last = new_v
end

