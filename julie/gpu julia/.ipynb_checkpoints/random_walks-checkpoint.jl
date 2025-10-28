using AMDGPU 
using Random

function random_walk(T)
    x = Int32(0)
    for _ in 1:T
        x += (rand(Float32) < 0.5f0) * Int32(2) - Int32(1)
    end
    return x 
end 

function walks(N, T) 
    walks = Vector{Int32}(undef, N);
    walks .= random_walk.(T)
    return walks
end 

function walks_gpu(N, T); 
    walks = ROCArray{Int32}(undef, N); 
    walks .= random_walk.(T) 
    return walks 
end

N = 2048;
T = 100; 
xs = walks(N, T) 
xs_gpu = walks_gpu(N, T) 
@show reduce(+, xs) / length(xs) 
@show reduce(+, xs_gpu) / length(xs_gpu) 

# This is slower than using fused kernel 
function random_walk!(x, dx, T) 
    fill!(x, zero(eltype(x))) 
    for _ in 1:T 
        Random.rand!(dx) 
        x .+= (dx .< 0.5f0) .* Int32(2) .- Int32(1)
    end 
    return x
end 

function walks_gpu_array(N, T) 
    walks = ROCArray{Int32}(undef, N); 
    dx = ROCArray{Float64}(undef, N); 
    random_walk!(walks, dx, T);
    return walks 
end 

