module plate
using SmoothedParticles
using Parameters
import LinearAlgebra
import StaticArrays
include("tools.jl")

#DECLARE CONSTANTS
#-----------------

const L = 0.06
const W = 0.01
#const L = 0.35               #length of the tail
#const W = 0.02               #width of the tail
#const c_0 = 45.709 #longitudinal sound speed
#const c_s = 16.976 #shear sound speed
const c_s = 9046.59
const c_0 = c_s
const c = sqrt(c_0^2 + 4/3*c_s^2)
const c_p = 4.0*c_0#0.
const g = RealVector(0.,-10.,0.)

const rho0 = 1845.0
#const rho0 = 1.0

const dr = W/40
#const dr = 3.9e-3
const h = 3.0001dr
#const h = 2.0*dr
const m0 = rho0*dr*dr

const dt = 0.05*dr/c
const t_end = 1e-5#3e-5
#const t_end = 1.0
const dt_plot = max(t_end/50, dt)

function init_velocity(x::RealVector)::RealVector
    A = 4.3369e-5
    omega = 2.3597e5
    alpha = 78.834
    a1 = 56.6368
    a2 = 57.6455
    s = alpha*(x[1] + L/2)
    v = A*omega*(a1*(sinh(s) + sin(s)) - a2*(cosh(s) + cos(s)))
    return 0*v*VECY
end

const inv = StaticArrays.inv

#DECLARE VARIABLES
#-----------------

@with_kw mutable struct Particle <: AbstractParticle
	m::Float64  = m0                  #mass
    x::RealVector                     #position
    X::RealVector
    v::RealVector = init_velocity(x)  #velocity
    #v::RealVector = VEC0  #velocity

    P::Float64 = 0.0      #pressure
    f::RealVector = VEC0  #force
    A::RealMatrix = MAT0
    #MAT1  #distortion
    T::RealMatrix = MAT0  #stress
    L::RealMatrix = MAT0
    Pmat::RealMatrix = MAT0
    Qmat::RealMatrix = MAT0
    
    rho::Float64 = 0.
    lambda::Float64 = 0.

    C_rho::Float64 = 0.
    C_lambda::Float64 = 0.
end

#CREATE INITIAL STATE
#--------------------

function make_geometry()
    #grid = Grid(dr, :hexagonal)
    grid = Grid(dr, :square)
    rod = Rectangle(-L/2, -W/2, L/2, W/2)
    dom = BoundaryLayer(rod, grid, L + W)
    sys = ParticleSystem(Particle, dom, h)
    generate_particles!(sys, grid, rod, x -> Particle(x=x, X=x))
    create_cell_list!(sys)
    apply!(sys, find_rho!)
    for p in sys.particles
        p.C_rho = rho0 - p.rho
        p.C_lambda = -p.lambda
    end
    apply!(sys, reset!)
    apply_ternary!(sys, find_A_new!)
    apply!(sys, update_A_new!)
    apply!(sys, find_rho!)
    apply!(sys, find_T!)
    apply!(sys, find_f!)   
    return sys
end


#DECLARE PHYSICS
#---------------

function update_v!(p::Particle)
    p.v += 0.5*dt*p.f/p.m
    # p.v += 0.5*dt*g     # gravitational acceleration
    #  #dirichlet bc
    # if p.X[1] < h
    #     p.v = VEC0
    # end
end

function update_x!(p::Particle)
    p.x += 0.5*dt*p.v
end

function find_L!(p::Particle, q::Particle, r::Float64)
    ker = q.m*rDwendland2(h,r)
    x_pq = p.x-q.x
    v_pq = p.v-q.v
    p.T += ker*outer(x_pq, x_pq)
    p.L += ker*outer(v_pq, x_pq)
end

function update_A!(p::Particle)
    p.L = p.L*subinv(p.T)
    p.A = p.A*(MAT1 - 0.5*dt*p.L)*inv(MAT1 + 0.5*dt*p.L)
end

# This is the new particle based definition of A 
function find_A_new!(p::Particle, q::Particle, r::Particle, rq::Float64, rr::Float64) 
    # ker should probably be normalized
    ker = wendland2(h,rq)*wendland2(h,rr) 
    #p.normalizer += ker
    x_qr = q.x - r.x 
    X_qr = q.X - r.X 
    p.Pmat -= ker*outer(X_qr, x_qr) 
    p.Qmat -= ker*outer(x_qr, x_qr)
end

function update_A_new!(p::Particle)
    p.A = p.Pmat*subinv(p.Qmat)
end

function find_A_old!(p::Particle, q::Particle, r::Float64) 
    ker = wendland2(h, r) 
    x_pq = p.x - q.x 
    X_pq = p.X - q.X
    p.Pmat += ker*outer(X_pq, x_pq)
    p.Qmat += ker*outer(x_pq, x_pq)
end

function reset_A!(p::Particle) 
    p.Pmat = MAT0
    p.Qmat = MAT0
end 

function find_rho!(p::Particle, q::Particle, r::Float64)
    x_pq = p.x-q.x
    p.T += q.m*rDwendland2(h,r)*outer(x_pq, x_pq)
    p.rho += q.m*wendland2(h,r)
    p.lambda += q.m*wendland2h(h,r)
end

function find_T!(p::Particle)
    G = transpose(p.A)*p.A
    p.P = c_0^2*(p.rho - rho0)*rho0/p.rho
    p.T = -p.P/p.rho^2*MAT1 + c_s^2*G*dev(G)*subinv(p.T)
end

function find_f!(p::Particle, q::Particle, r::Float64)
    ker = q.m*rDwendland2(h,r)
    kerh = q.m*rDwendland2h(h,r)
    x_pq = p.x-q.x
    #stress
    p.f += p.m*ker*(p.T + q.T)*x_pq
    #anti-clumping force
    p.f += -p.m*kerh*(c_p/rho0)^2*(p.lambda + q.lambda)*x_pq
end

function reset!(p::Particle)
    p.f = VEC0
    p.L = MAT0
    p.T = MAT0
    p.rho = p.C_rho
    p.lambda = p.C_lambda
    # reset temporary P and Q matrices
    #p.Qmat = MAT0
    #p.Pmat = MAT0
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


#ENERGIES
#--------

function pE_kinetic(p::Particle)::Float64
    return 0.5*p.m*dot(p.v, p.v)
end 

function pE_bulk(p::Particle)::Float64
    return 0.5*p.m*c_0^2*(p.rho - rho0)^2/p.rho^2
end

function pE_shear(p::Particle)::Float64
    G = transpose(p.A)*p.A
    return 0.25*p.m*c_s^2*LinearAlgebra.norm(dev(G),2)^2
end

function pE_penalty(p::Particle)::Float64
    return 0.5*p.m*c_p^2*(p.lambda/rho0)^2
end


#TIME ITERATION
#--------------

function main()
    sys = make_geometry()
    centre = find_minimizer(p -> LinearAlgebra.norm(p.x), sys)
    out = new_pvd_file("results/plate")
    csv_data = open("results/plate/data.csv", "w")
    write(csv_data, string("t,y,E_total,E_kinetic,E_bulk,E_shear,E_penalty\n"))
    @time for k = 0 : Int64(round(t_end/dt))
        t = k*dt
        if (k % Int64(round(dt_plot/dt)) == 0)
            @show t
            println("N = ", length(sys.particles)) 
            y = centre.x[2]
            E_kinetic = sum(p -> pE_kinetic(p), sys.particles)
            E_bulk = sum(p -> pE_bulk(p), sys.particles)
            E_shear = sum(p -> pE_shear(p), sys.particles)
            E_penalty = sum(p -> pE_penalty(p), sys.particles)
            E_total = E_kinetic + E_bulk + E_shear + E_penalty
            @show E_total
            @show E_kinetic
            @show E_bulk
            @show E_shear
            @show E_penalty
            println()
            write(csv_data, vec2string([t, y, E_total, E_kinetic, E_bulk, E_shear, E_penalty]))
            save_frame!(out, sys, :v, :A, :P, :Pmat, :Qmat)
        end
        apply!(sys, update_v!)
        apply!(sys, update_x!)
        create_cell_list!(sys)
        apply!(sys, reset!)
        apply!(sys, reset_A!)
        #apply!(sys, find_L!)
        #apply!(sys, update_A!)
        #apply_ternary!(sys, find_A_new!)
        apply!(sys, find_A_old!)
        apply!(sys, update_A_new!)
        apply!(sys, update_x!)
        create_cell_list!(sys)
        apply!(sys, reset!)
        apply!(sys, find_rho!)
        apply!(sys, find_T!)
        apply!(sys, find_f!)        
        apply!(sys, update_v!)
    end
    save_pvd_file(out)
    close(csv_data)
end

end #module
