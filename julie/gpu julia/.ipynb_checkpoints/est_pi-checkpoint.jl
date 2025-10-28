using AMDGPU 
AMDGPU.allowscalar(false)
using BenchmarkTools 

N = 256;
a = AMDGPU.rand(N);
b = AMDGPU.rand(N); 
c = similar(a);

typeof(a) <: AbstractArray

function my_add!(c::AbstractArray, a::AbstractArray, b::AbstractArray) 
    for i in eachindex(c) 
        c[i] = a[i] + b[i]
    end 
    nothing 
end 

my_add!(c, a, b) 

# Check if the addition is correct 
isapprox(Array(c), Array(a) .+ Array(b))

