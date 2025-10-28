using SmoothedParticles
using Parameters
using StaticArrays

const dr = 0.01
const h = (1.0)dr
#const h = (1.0 + 10.0*eps(Float64))dr

@with_kw mutable struct Particle <: AbstractParticle
    x::RealVector 
    a::Float64 = 1.0
    b::Float64 = 0.0
end 

function make_geometry()
    grid = Grid(dr, :square)
    dom = Rectangle(-1.0, -1.0, 1.0, 1.0)
    sys = ParticleSystem(Particle, dom, h)
    generate_particles!(sys, grid, dom, x -> Particle(x=x))
    create_cell_list!(sys)
    return sys
end

function test!(p::Particle, q::Particle, r::Particle, rq::Float64, rr::Float64)
    p.b += q.a * r.a
end

function add_a!(p::Particle, q::Particle, r::Float64)
   #p.b += q.a 
   p.b += 1.0
end

function dist2(p::AbstractParticle, q::AbstractParticle)::Float64
	return dot(p.x - q.x, p.x - q.x)
end

@inline function _apply_ternary!(sys::ParticleSystem, action!::Function, p::AbstractParticle)
    key = SmoothedParticles.find_key(sys, p.x)
    # collect neighbours of p
    neighbours = Int[]
    for Δkey in sys.key_diff
        neigh_key = key + Δkey
        if 1 <= neigh_key <= sys.key_max
            for j in sys.cell_list[neigh_key].entries
                if j == 0; break; end
                q = sys.particles[j]
                if q === p; continue; end
                rq = dist(p, q) 
                if rq <= sys.h
                    push!(neighbours, j)
                end
            end
        end
    end
    # For each neighbour q, loop over all neighbours r 
    for k in neighbours 
        #if k == 0; break; end 
        q = sys.particles[k] 
        rq = dist(p, q) 
        for l in neighbours 
            #if l == 0; break; end 
            r = sys.particles[l]
            if r == p || r == q; continue; end 
            rr = dist(p, r) 
            action!(p, q, r, rq, rr)
        end 
    end 
end

@inline function _apply_ternary_copy!(sys::ParticleSystem, action!::Function, p::AbstractParticle)
    key = SmoothedParticles.find_key(sys, p.x)
    # collect neighbours of p
    neighbours = Int[]
    distances = Float64[]
    for Δkey in sys.key_diff
        neigh_key = key + Δkey
        if 1 <= neigh_key <= sys.key_max
            for j in sys.cell_list[neigh_key].entries
                if j == 0; break; end
                q = sys.particles[j]
                if q === p; continue; end
                rq = dist(p, q) 
                if rq <= sys.h
                    push!(neighbours, j)
                    push!(distances, rq)
                end
            end
        end
    end
    # For each neighbour q, loop over all neighbours r 
    for k in eachindex(neighbours) 
        #if k == 0; break; end 
        q = sys.particles[neighbours[k]] 
        #rq = dist(p, q) 
        rq = distances[k]
       for l in eachindex(neighbours)
            #if l == 0; break; end 
            r = sys.particles[neighbours[l]]
            if r == p || r == q; continue; end 
            #rr = dist(p, r) 
            rr = distances[l]
            action!(p, q, r, rq, rr)
        end 
    end 
end

@inline function _apply_ternary_fast!(sys::ParticleSystem, action!::Function, p::AbstractParticle)
    key = SmoothedParticles.find_key(sys, p.x)
    h2 = sys.h^2
    # collect neighbours of p
    neighbours = Int[]
    for Δkey in sys.key_diff
        neigh_key = key + Δkey
        if 1 <= neigh_key <= sys.key_max
            for j in sys.cell_list[neigh_key].entries
                if j == 0; break; end
                q = sys.particles[j]
                if q === p; continue; end
                rq2 = dist2(p, q) 
                if rq2 <= h2
                    push!(neighbours, j)
                end
            end
        end
    end
    # For each neighbour q, loop over all neighbours r 
    for k in neighbours 
        #if k == 0; break; end 
        q = sys.particles[k] 
        rq = dist(p, q) 
        for l in neighbours 
            #if l == 0; break; end 
            r = sys.particles[l]
            if r == p || r == q; continue; end 
            rr = dist(p, r) 
            action!(p, q, r, rq, rr)
        end 
    end 
end

@inline function _apply_ternary_prealloc!(sys::ParticleSystem, action!::Function, p::AbstractParticle)
    # Estimate maximum number of neighbors in 2D
    N_max = Int(ceil(pi * sys.h^2 / dr^2))
    #neighbours = Vector{Int}(undef, N_max)
    # this does not work, because the Vector contains garabe values
    #neighbours = zeros(Int, N_max)
    neighbours = MVector{N_max, Int}(zeros(Int, N_max))
    ncount = 0
    key = SmoothedParticles.find_key(sys, p.x)
    # collect neighbours of p
    #neighbours = Int[]
    for Δkey in sys.key_diff
        neigh_key = key + Δkey
        if 1 <= neigh_key <= sys.key_max
            for j in sys.cell_list[neigh_key].entries
                if j == 0; break; end
                q = sys.particles[j]
                if q === p; continue; end
                rq = dist(p, q) 
                if rq <= sys.h
                    ncount += 1
                    #push!(neighbours, j)
                    neighbours[ncount] = j
                end
            end
        end
    end
    # For each neighbour q, loop over all neighbours r 
    for k in neighbours 
        if k == 0; break; end 
        q = sys.particles[k] 
        rq = dist(p, q) 
        for l in neighbours 
            if l == 0; break; end 
            r = sys.particles[l]
            if r == p || r == q; continue; end 
            rr = dist(p, r) 
            action!(p, q, r, rq, rr)
        end 
    end 
end

function apply_ternary!(sys::ParticleSystem, action!::Function)
    Threads.@threads for p in sys.particles
        _apply_ternary_prealloc!(sys, action!, p)
    end
end

function apply_ternary!(sys::ParticleSystem, action!::Function)
    Threads.@threads for p in sys.particles
        _apply_ternary!(sys, action!, p)
        #_apply_ternary_prealloc!(sys, action!, p)
        #_apply_ternary_copy!(sys, action!, p)
        #_apply_ternary_fast!(sys, action!, p)
    end 
end 

function main()
    out = new_pvd_file("results")
    sys = make_geometry()
    #apply_ternary!(sys, test!)
    apply!(sys, add_a!)
    save_frame!(out, sys, :a, :b)
    save_pvd_file(out)
end 