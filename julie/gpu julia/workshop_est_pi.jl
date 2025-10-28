using AMDGPU
using BenchmarkTools

# OLD METHODS
function throw_dart()
    x = rand() * 2 - 1
    y = rand() * 2 - 1
    return (x^2+y^2<=1)
end 

function est_pi_gpu_array(N)
    darts = ROCArray{Bool}(undef, N)
    dart .= (_->throw_dart()).(nothing)
    est = 4 * reduce(+, darts, inti=0) / N 
    return est 
end

# NEW KERNEL 
function _est_pi_kernel(global_count)
    # create some shared memory for each thread in the block
    hits = AMDGPU.alloc_local(:hits, UInt16, 256)
    # hits = CUDA.@cuStaticShareMem(UInt16, 256)
    
    # throw a dart and calculate whether it has hit 
    x = rand() * 2 - 1
    y = rand() * 2 - 1
    is_hit = (x^2+y^2<=1)

    idx = AMDGPU.threadIdx().x
    # record the hit in the shared memory
    # hits[idx] = UInt16(is_hit)
    hits[idx] = UInt16(is_hit) 

    # perform a reduction on the shared memory
    step_size = 256 ÷ 2 
    while (step_size != 0)
        # CUDA.sync_threads()
        AMDGPU.sync_workgroup()
        if (idx <= step_size)
            hits[idx] += hits[idx+step_size]
        end
        step_size ÷= 2
    end

    # add the count from the block into the global count
    if idx == 1
        # CUDA.@atomic global_count[] += hits[1]
        AMDGPU.atomic_add!(global_count[], 1, hits[1])
    end

    return nothing
end

function est_pi(n)
    # Calculate number of threads and blocks
    threads = 256
    blocks = cld(n, threads)

    # Create some memory to store the count 
    total_count = AMDGPU.zeros(UInt32, 1)
    # Run the KERNEL 
    @roc gridsize=blocks groupsize=threads _est_pi_kernel(total_count) 
    # Transfer the finished count from the GPU
    count = Array(total_count)[1]
    # count = UInt32(0)
    # CUDA.@allowscalar begin 
    #     count = total_count[]
    # end 
    # # CUDA.unsafe_free!(total_count)
    return 4 * count / (blocks * threads)
end

