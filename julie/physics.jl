using BenchmarkTools
using Test
using StaticArrays
include("initial_conditions.jl")

function test_benchmark()
    N = 100
    positions, velocities, masses = initial_conditions(N, 1, 5)
    G = 5.0
    dt = 0.01

    update_cache = init_update_cache(N)

    original_next_pos, original_next_vel = update_positions(positions, velocities, masses, G, dt)
    optimised_next_pos, optimized_next_vel = update_positions_fast!(update_cache, positions, velocities, masses, G, dt)
    @testset "Positions" begin
        @test original_next_pos ≈ optimised_next_pos
    end
    @testset "Velocities" begin
        @test original_next_vel ≈ optimized_next_vel
    end
    display(@benchmark update_positions($positions, $velocities, $masses, $G, $dt))
    display(@benchmark update_positions_fast!($update_cache, $positions, $velocities, $masses, $G, $dt))
end

function update_positions(positions, velocities, masses, G, dt)
    # Calculate next position based on Runge Kutta method
    k1v = dt * acceleration(positions, masses, G)
    k1p = dt * velocities

    k2v = (dt / 2) * acceleration(positions + 0.5 * k1p, masses, G)
    k2p = (dt / 2) * (velocities + 0.5 * k1v)

    k3v = (dt / 2) * acceleration(positions + 0.5 * k2p, masses, G)
    k3p = (dt / 2) * (velocities + 0.5 * k2v)

    k4v = dt * acceleration(positions + k3p, masses, G)
    k4p = dt * (velocities + k3v)

    next_velocities = velocities + (k1v + 2 * k2v + 2 * k3v + k4v) / 6
    next_positions = positions + (k1p + 2 * k2p + 2 * k3p + k4p) / 6

    return next_positions, next_velocities
end

function acceleration(positions, masses, G)
    # Positions: N x 3 matrix with x, y, z coordinates
    # Masses: N vector
    # G: Gravitational constant

    N = size(positions, 1)
    # Vector (N)
    x = positions[:, 1]
    y = positions[:, 2]
    z = positions[:, 3]

    # Create a matrix of distances between particles
    # Matrix (NxN)
    dx = (x' .- x)
    dy = (y' .- y)
    dz = (z' .- z)

    r = sqrt.(dx .^ 2 + dy .^ 2 + dz .^ 2)
    r = max.(r, 1e-3) # Avoid division by zero

    F = G * (masses .* masses') ./ (r .^ 2)
    # Set the diagonal to zero to avoid self-interaction
    for i in 1:N
        F[i, i] = 0
    end

    # Calculate force components for each pair of particles
    Fx = F .* dx ./ r
    Fy = F .* dy ./ r
    Fz = F .* dz ./ r

    # Calculate net force 
    Fx = sum(Fx, dims=2)
    Fy = sum(Fy, dims=2)
    Fz = sum(Fz, dims=2)

    # Use F=ma to calculate the acceleration
    A = hcat(Fx, Fy, Fz) ./ masses
    return A
end

function init_update_cache(N, T=Float64)
    return(;
        k1v=Matrix{T}(undef, N, 3),
        k1p=Matrix{T}(undef, N, 3),
        k2v=Matrix{T}(undef, N, 3),
        k2p=Matrix{T}(undef, N, 3),
        k3v=Matrix{T}(undef, N, 3),
        k3p=Matrix{T}(undef, N, 3),
        k4v=Matrix{T}(undef, N, 3),
        k4p=Matrix{T}(undef, N, 3),
        next_positions=Matrix{T}(undef, N, 3),
        accelerations=Matrix{T}(undef, N, 3),
        next_velocities=Matrix{T}(undef, N, 3)
    )
end

function update_positions_fast!(cache, positions, velocities, masses, G, dt)
    # Calculate next position based on Runge Kutta method
    cache.k1v .= dt .* acceleration_fast!(cache.accelerations, positions, masses, G)
    cache.k1p .= dt .* velocities

    cache.next_positions .= @. positions + 0.5 * cache.k1p
    cache.k2v .= (dt / 2) .* acceleration_fast!(cache.accelerations, cache.next_positions, masses, G)
    cache.k2p .= @. (dt / 2) * (velocities + 0.5 * cache.k1v)

    cache.next_positions .= @. positions + 0.5 * cache.k2p
    cache.k3v .= (dt / 2) .* acceleration_fast!(cache.accelerations, cache.next_positions, masses, G)
    cache.k3p .= @. (dt / 2) * (velocities + 0.5 * cache.k2v)

    cache.next_positions .= positions .+ cache.k3p
    cache.k4v .= dt .* acceleration_fast!(cache.accelerations, cache.next_positions, masses, G)
    cache.k4p .= @. dt * (velocities + cache.k3v)

    cache.next_velocities .= @. velocities + (cache.k1v + 2 * cache.k2v + 2 * cache.k3v + cache.k4v) / 6
    cache.next_positions .= @. positions + (cache.k1p + 2 * cache.k2p + 2 * cache.k3p + cache.k4p) / 6

    return cache.next_positions, cache.next_velocities
end

function acceleration_fast!(accelerations, positions, masses, G)
   N, D = size(positions)
   accelerations .= 0
   for i in 1:N
       for j in 1:N
           if i == j
               continue
           end
           # (3, ) Vector
           #p_i = SVector(positions[i,1], positions[i,2], positions[i,3])
           #p_j = SVector(positions[j,1], positions[j,2], positions[j,3])
           p_i = @SVector [positions[i, k] for k in 1:3]
           p_j = @SVector [positions[j, k] for k in 1:3]

           # (3, ) Vector
           dp_ij = p_j .- p_i

           # Scalars
           r_ij = sqrt(sum(x->x^2, dp_ij))
           F_ij = G * (masses[i] * masses[j]) / (r_ij^2)
           # (3, ) Vector
           a_ij = (F_ij / masses[i] / r_ij) .* dp_ij   
           #accelerations[i, :] .+= a_ij        #allocates a lot of memory
           # Use a loop to avoid allocation
           for k in 1:D
                accelerations[i, k] += a_ij[k]
           end      
       end
   end
   return accelerations
end