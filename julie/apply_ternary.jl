using SmoothedParticles
using Parameters
using StaticArrays

const dr = 0.01

const h = 1.0dr
const h_eff = (1.0 + 10.0*eps(Float64))dr

@with_kw mutable struct Particle <: AbstractParticle
    x::RealVector 
    number_of_neighbours::Float64 = 0.0              #This stores the number of neighbours within sys.h
    number_of_neighbour_pairs::Float64 = 0.0         #This stores the number of neighbour pairs within sys.h
end 

function make_geometry(h)
    grid = Grid(dr, :square)
    dom = Rectangle(-1.0, -1.0, 1.0, 1.0)
    sys = ParticleSystem(Particle, dom, h)
    generate_particles!(sys, grid, dom, x -> Particle(x=x))
    create_cell_list!(sys)
    return sys
end

function add_one!(p::Particle, q::Particle, r::Float64)
   p.number_of_neighbours += 1.0
end

function add_one!(p::Particle, q::Particle, r::Particle, rq::Float64, rr::Float64)
    p.number_of_neighbour_pairs += 1.0
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
                # This line might cause problems in floating point arithmetic
                # (as well as in _apply_binary!) 
                # a solution would be to write:
                # if rq <= sys.h + EPS
                # or at least mention it in documentatation
                if rq <= sys.h
                    push!(neighbours, j)
                end
            end
        end
    end
    # For each neighbour q, loop over all neighbours r 
    for k in neighbours 
        q = sys.particles[k] 
        rq = dist(p, q) 
        for l in neighbours 
            r = sys.particles[l]
            if r == p || r == q; continue; end 
            rr = dist(p, r) 
            action!(p, q, r, rq, rr)
        end 
    end 
end

"""
    apply_ternary!(sys::ParticleSystem, action!::Function)

Apply a ternary operator `action!(p::T, q::T, r::T, 
                                  rq::Float64, rr::Float64)` 
between particle `p` and all pairs of its neighbours `q` and `r`
in `sys::ParticleSystem{T}`. Values `rq` and `rr` are the
distances between `p` and `q` and between `p` and `r`, respectively.
This excludes all particles `q` and `r` with distance greater than `sys.h`.
This has complexity O(N*k^2) where N is the number of particles and k the
average number of neighbours per particle and runs in parallel.

!!! warning "Warning"
    Modifying particles `q` or `r` within `action!` can lead to race condition.
    Selecting large `sys.h` leads to significant performance drop.
"""
function apply_ternary!(sys::ParticleSystem, action!::Function)
    Threads.@threads for p in sys.particles
        _apply_ternary!(sys, action!, p)
    end 
end 

function main()
    out_h = new_pvd_file("results/h")
    out_h_eff = new_pvd_file("results/h_eff")

    #Create two systems one using h and one using h_eff
    sys_h = make_geometry(h)
    sys_h_eff = make_geometry(h_eff)

    apply_ternary!(sys_h, add_one!)
    apply!(sys_h, add_one!)

    apply_ternary!(sys_h_eff, add_one!)
    apply!(sys_h_eff, add_one!)
 
    save_frame!(out_h, sys_h, :number_of_neighbours, :number_of_neighbour_pairs)
    save_pvd_file(out_h)
    
    save_frame!(out_h_eff, sys_h_eff, :number_of_neighbours, :number_of_neighbour_pairs)
    save_pvd_file(out_h_eff)
 
end 