using MPI
MPI.Init() 

@inline function throw_dart()
    x = rand() * 2 - 1 
    y = rand() * 2 - 1 
    return x^2 + y^2 <= 1 
end

function count_hits(n)
    hits = 0 
    @simd for _ in 1:n 
        hits += throw_dart() 
    end 
    return hits 
end

function main() 
    comm = MPI.COMM_WORLD
    comm_size = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm) 

    N = 10^8 
    chunksize = cld(N, comm_size) 
    hits = count_hits(chunksize) 

    println("Rank $rank, hits: $hits")
    total_hits = MPI.Reduce(hits, MPI.SUM, 0, comm)
    MPI.Barrier(comm)
    println("Rank $rank, total_hits: $total_hits")

    if rank == 0 
        println("Running final statement")
        pi_estimate = 4 * total_hits / N
        println("Pi estimate: $pi_estimate using $N darts")
    end 
end

main()