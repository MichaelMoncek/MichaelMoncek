using StaticArrays

const RealVector = SArray{Tuple{3}, Float64, 1, 3}
const Real4Vector = SArray{Tuple{4}, Float64, 1, 4}

function dot(x::RealVector, y::RealVector)::Float64 
    @inbounds return x[1]*y[1] + x[2]*y[2] + x[3]*y[3]
end

function norm(x::RealVector)::Float64
    return sqrt(dot(x, x))
end

function norm2(x::RealVector)::Float64
    return dot(x, x)
end
