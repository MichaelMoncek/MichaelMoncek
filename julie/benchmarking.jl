# struct Vector2D{T<:Real}
#     x::T
#     y::T
# end

# function Base.:+(a::Vector2D, b::Real)
#     return Vector2D(a.x + b, a.y + b)
# end
# Base.:+(a::Real, b::Vector2D) = b + a
# Base.:+(a::Vector2D, b::Vector2D) = Vector2D(a.x + b.x, a.y + b.y)
# Base.:-(a::Vector2D, b::Real) = a + (-b)
# # v1 = Vector2D(5,6)
# # v2 = Vector2D{Int64}(5.0, 6)
# # v3 = Vector2D("1","2")

# g(x) = "default"
# g(x::Integer) = "abstract integer"
# g(x::Int) = "integer"

# f(x::Real) = 2x^2 -3x + 5

# h(x, y, z) = "default $x $y $z"
# h(x, y, z::Integer) = "last integer $x $y $z"
# h(x, y::Integer, z::Integer) = "last two integer $x $y $z"
# h(x::Float64, y, z::Int) = "first float, last int $x $y $z"

# x = [1;; 2;; 3]
# x = [5, 9, 2]

# y = [n*n for n in x]
# y = map(n->n*n, x)
# y = x .* x
# f(n) = n*n
# y = f.(x)
# y = (n-> n*n).(x)
# y = x .|> f
# y = zeros(Int, 3)
# s = 0
# for i in eachindex(x)
#     n = x[i]
#     y[i] = n*n
# end
# y

# x = (5, 9, "Hello there")
# typeof(x)
# y = (n->n*n).(x[1:2])

# z = (1, 2, y...)

# function sumsquares(array...)
#     s = 0
#     for n in array
#         s += n*n
#     end
#     return s, length(array)
# end

# s, _ = sumsquares(2,3,4)
# s
# num

#@time rand(1000,1000);

function vectoradd(a, b)
    c = similar(a)
    @assert length(a) == length(b) 
    for i in eachindex(a, b)
        c[i] = a[i] + b[i]
    end
    return c
end

function vectoradd!(c, a, b)
    @assert length(a) == length(b) 
    for i in eachindex(a, b)
        c[i] = a[i] + b[i]
    end
    return c
end
#ai = rand(1000);
#bi = rand(1000);
#@benchmark vectoradd(ai, bi)
#c = similar(ai);
#@benchmark vectoradd!(c, ai, bi)