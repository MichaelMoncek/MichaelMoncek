using AMDGPU
using BenchmarkTools 

function throw_dart()
    x = rand() * 2 - 1
    y = rand() * 2 - 1
    return (x^2+y^2<=1)
end

function est_pi(N)
    hits = mapreduce(_->throw_dart(), +, 1:N);
    return 4 * hits / N 
end 

function est_pi_gpu(N)
    darts = ROCArray{Bool}(undef, N)
    darts .= (_->throw_dart()).(nothing) # This is done to call multiple instances of throw_dart
    est = 4 * reduce(+, darts, init=0) / N 
    # CUDA.unsafe_free(darts) # there isn't AMD equivallent as 
    # AMD relies on Julia's GC to trigger freeing of memory
    return est
end    

@benchmark est_pi($(2^20))
@benchmark AMDGPU.@sync est_pi_gpu($(2^20))
