using Base.Threads

function pi_estimate(n)
    hits = 0
    for _ in 1:n 
        x = rand() * 2 - 1 
        y = rand() * 2 - 1 
        is_hit = x^2 + y^2 <= 1 
        hits += is_hit 
    end 
    return 4 * hits / n 
end

function pi_estimate_atomics(n)
    hits = Atomic{Int}(0)
    @threads for _ in 1:n 
        x = rand() * 2 - 1 
        y = rand() * 2 - 1 
        is_hit = x^2 + y^2 <= 1 
        # 1. Loads hits 
        # 2. Load is_hit 
        # 3. Add hits + is_hit 
        # 4. Store result in hits
        atomic_add!(hits, Int(is_hit))
    end 
    return 4 * hits[] / n 
end

function pi_estimate_mutexes(n)
    hits = 0
    lck = ReentrantLock()
    @threads for _ in 1:n 
        x = rand() * 2 - 1 
        y = rand() * 2 - 1 
        is_hit = x^2 + y^2 <= 1 
        lock(lck) do
            hits += is_hit 
        end
    end 
    return 4 * hits / n 
end

function pi_estimate_semaphore(n)
    hits = 0
    hit_pool = Channel{Int}(Threads.nthreads())
    for _ in 1:Threads.nthreads()
        put!(hit_pool, 0)
    end
    @threads for _ in 1:n 
        x = rand() * 2 - 1 
        y = rand() * 2 - 1 
        is_hit = x^2 + y^2 <= 1 
        hits = take!(hit_pool)
        hits += is_hit 
        put!(hit_pool, hits)
    end 
    total_hits = sum(take!(hit_pool) for _ in 1:Threads.nthreads())
    return 4 * total_hits / n
end

function pi_estimate_threads(n)
    hits = zeros(Int, Threads.nthreads())
    @threads for _ in 1:n 
        x = rand() * 2 - 1 
        y = rand() * 2 - 1 
        is_hit = x^2 + y^2 <= 1 
        hits[Threads.threadid()] += is_hit 
    end
    return 4 * sum(hits) / n 
end

function pi_estimate_threads_chunks(n)
    total_hits = Atomic{Int}(0)
    W = Threads.nthreads()
    chunk_size = cld(n, W)
    @threads for _ in 1:W 
        hits = 0
        for _ in 1:chunk_size
            x = rand() * 2 - 1 
            y = rand() * 2 - 1 
            is_hit = x^2 + y^2 <= 1 
            hits += is_hit
        end
        atomic_add!(total_hits, hits)
    end 
    return 4 * total_hits[] / n
end