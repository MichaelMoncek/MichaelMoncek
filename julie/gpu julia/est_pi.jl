using AMDGPU 
AMDGPU.allowscalar(false)
using BenchmarkTools 

N = 2050;
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

function _my_add_kernel!(c, a, b)
    i = AMDGPU.threadIdx().x + (workgroupIdx().x - 1) * workgroupDim().x
    if i <= length(c)
        c[i] = a[i] + b[i]
        return nothing
    end 
end 

function my_add!(c::ROCArray, a::ROCArray, b::ROCArray)
    @roc gridsize=cld(length(c), 1024) groupsize=1024 _my_add_kernel!(c, a, b)
    return nothing
end
my_add!(c, a, b) 

# Check if the addition is correct 
isapprox(Array(c), Array(a) .+ Array(b))

