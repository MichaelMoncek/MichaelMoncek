module ldc

using SmoothedParticles
using Parameters
import LinearAlgebra
import StaticArrays
include("tools.jl")

#DECLARE CONSTANTS
#-----------------

const llid = 1.0                #length of the lid
const vlid = 1.0				#flow speed of the lid
const rho0 = 1.0                #density

const Re = 100.0
const c_s = 20.0
const c_0 = 20.0

const nu = vlid*llid/Re
const tau = 6*nu/c_s^2
const dr = llid/200
const h = 2.0dr
const m0 = rho0*dr*dr
const wwall = 2h
const c_p = 0.
const c = sqrt(c_0^2 + 4/3*c_s^2)

const dt = min(0.1*dr/c, 0.1tau)
const t_end = 10.0
const dt_plot = max(t_end/200, dt)
const t_acc = 0.5

const A_alpha = 0.05*dt

const inv = StaticArrays.inv
const det = StaticArrays.det

#PARTICLE FLAGS
const FLUID = 0.
const WALL = 1.
const LID = 2.

#DECLARE VARIABLES
#-----------------

@with_kw mutable struct Particle <: AbstractParticle
	m::Float64  = m0                  #mass
    x::RealVector                     #position
    v::RealVector = VEC0  #velocity
    P::Float64 = 0.0      #pressure
    A::RealMatrix = MAT1  #distortion
    T::RealMatrix = MAT0  #stress
    L::RealMatrix = MAT0
    A_::RealMatrix = MAT0
    f::RealVector = VEC0
    rho::Float64 = 0.
    C_rho::Float64 = 0.
    type::Float64
end

function compute_fluxes(sys::ParticleSystem, res = 100)
    s = range(0.,1.,length=res)
    fluxes = open("results/ldc/fluxes.csv", "w")
    write(fluxes, "s,v1,v2\n")
    for i in 1:res
		#x-velocity along y-centerline
		x = RealVector(0.5, s[i], 0.)
		gamma = SmoothedParticles.sum(sys, (p,r) -> p.m*wendland2(h,r), x)
        v1 = SmoothedParticles.sum(sys, (p,r) -> p.m*p.v[1]*wendland2(h,r), x)/gamma
		#y-velocity along x-centerline
		x = RealVector(s[i], 0.5, 0.)
		gamma = SmoothedParticles.sum(sys, (p,r) -> p.m*wendland2(h,r), x)
        v2 = SmoothedParticles.sum(sys, (p,r) -> p.m*p.v[2]*wendland2(h,r), x)/gamma
		#save results into csv
        write(fluxes, vec2string([s[i],v1, v2]))
    end
    close(fluxes)
end

#CREATE INITIAL STATE
#--------------------

@fastmath function dist_from_dom(x::RealVector)::Float64
    d1 = abs(x[1] - 0.5) - 0.5llid
    d2 = abs(x[2] - 0.5) - 0.5llid
    return max(d1, d2, 0.)
end

@fastmath function corner_char(x::RealVector)::Bool
    lo = 2h
    hi = llid - lo
    return (max(lo-x[1], x[1]-hi, 0.)^2 + max(x[2]-hi, 0.)^2 > lo^2) && !(lo < x[1] < hi)
end

function make_geometry()
    grid = Grid(dr, :square)
    box = Rectangle(0., 0., llid, llid)
    dom = Rectangle(-wwall, -wwall, llid + wwall, llid + wwall)
    fluid = Specification(box, x -> !corner_char(x))
    walls = dom - fluid
    lid = Specification(walls, x -> x[2] > llid)
    wall = walls - lid
    lid = Specification(lid, x -> x[1] < llid + wwall - 0.4dr)
    fluid = box - walls
    sys = ParticleSystem(Particle, dom, h)
    generate_particles!(sys, grid, fluid, x -> Particle(x=x, type=FLUID))
    generate_particles!(sys, grid, lid, x -> Particle(x=x, type=LID))
    generate_particles!(sys, grid, wall, x -> Particle(x=x, type=WALL))
    create_cell_list!(sys)
    apply!(sys, find_rho!)
    for p in sys.particles
        p.C_rho = rho0 - p.rho
    end
    C_min = minimum(p -> p.C_rho, sys.particles)
    for p in sys.particles
        if p.type == LID
            p.C_rho = C_min
        end
    end
    apply!(sys, reset!)
    apply!(sys, find_rho!)
    apply!(sys, find_T!)
    apply!(sys, find_f!)   
    return sys
end

#DECLARE PHYSICS
#---------------

function update_v!(p::Particle, t::Float64)
    if p.type == FLUID
        p.v += 0.5*dt*p.f/p.m
    elseif p.type == LID
        p.v = (t > t_acc ? 1.0 : t/t_acc)*vlid*VECX
    end
end

function update_x!(p::Particle)
    p.x += 0.5*dt*p.v
    if p.x[1] > llid + wwall
        p.x = p.x - (llid + 2wwall)*VECX
    end
end

function find_L!(p::Particle, q::Particle, r::Float64)
    ker = q.m*rDwendland2(h,r)
    x_pq = p.x-q.x
    v_pq = p.v-q.v
    if q.type != FLUID
        d = dist_from_dom(q.x)
        v_pq = r*saveinv(r-d)*v_pq
    end
    p.T += ker*outer(x_pq, x_pq)
    p.L += ker*outer(v_pq, x_pq)
end

function update_A!(p::Particle)
    p.L = p.L*subinv(p.T)
    p.A = p.A*(MAT1 - 0.5*dt*p.L)*inv(MAT1 + 0.5*dt*p.L)
end

function find_rho!(p::Particle, q::Particle, r::Float64)
    x_pq = p.x-q.x
    p.T += q.m*rDwendland2(h,r)*outer(x_pq, x_pq)
    p.rho += q.m*wendland2(h,r)
end

function find_T!(p::Particle)
    G = transpose(p.A)*p.A
    p.P = c_0^2*(p.rho - rho0)*rho0/p.rho
    p.T = (p.type == FLUID)*c_s^2*G*dev(G)*subinv(p.T)
    p.A_ = p.A
    p.A = (1.0 - A_alpha)*p.A
end

function smoothing!(p::Particle, q::Particle, r::Float64)
    gamma = p.rho - p.C_rho + wendland2(h,0.)
    p.A += A_alpha*q.m/gamma*q.A_*wendland2(h,r)
end

function find_f!(p::Particle, q::Particle, r::Float64)
    ker = q.m*rDwendland2(h,r)
    x_pq = p.x-q.x
    #bulk force
    p.f += -p.m*ker*(p.P/p.rho^2 + q.P/q.rho^2)*x_pq
    #shear force
    p.f += p.m*ker*(p.T + q.T)*x_pq
    #NS viscosity
    #p.f += +2*p.m*ker*nu/rho0*(p.v - q.v)
end

function reset!(p::Particle)
    p.f = VEC0
    p.T = MAT0
    p.rho = p.C_rho
    p.f = VEC0
end

function reset_L!(p::Particle)
    p.L = MAT0
end

function relax_f(A)
    return -3/tau*A*dev(transpose(A)*A)
end

@inbounds function relax_A!(p::Particle)
    #RK4 scheme
    A_new = p.A
    K = relax_f(p.A)
    A_new += dt*K/6
    K = relax_f(p.A + dt*K/2)
    A_new += dt*K/3
    K = relax_f(p.A + dt*K/2)
    A_new += dt*K/3
    K = relax_f(p.A + dt*K)
    A_new += dt*K/6
    p.A = A_new
end


#TIME ITERATION
#--------------

function main()
    println("t1 = ", 0.2*dr/c)
    println("t2 = ", 0.1tau)
    sys = make_geometry()
    out = new_pvd_file("results/ldc")
    @time for k = 0 : Int64(round(t_end/dt))
        t = k*dt
        if (k % Int64(round(dt_plot/dt)) == 0)
            @show t
            N = length(sys.particles)
            @show N
            save_frame!(out, sys, :v, :A, :P, :type)
        end
        apply!(sys, p -> update_v!(p,t+0.5dt))
        apply!(sys, update_x!)
        create_cell_list!(sys)
        apply!(sys, reset!)
        apply!(sys, reset_L!)
        apply!(sys, find_L!)
        apply!(sys, update_A!)
        apply!(sys, relax_A!)
        apply!(sys, update_x!)
        create_cell_list!(sys)
        apply!(sys, reset!)
        apply!(sys, find_rho!)
        apply!(sys, smoothing!, self=true)
        apply!(sys, find_T!)
        apply!(sys, find_f!)        
        apply!(sys, p -> update_v!(p,t+dt))
    end
    save_pvd_file(out)
    close(csv_data)
    compute_fluxes(sys)
end

function test_benchmark(sys)    
    
    # function update_v_with_t!(p, t)
    #     (p -> update_v!(p, t))
    # end

    t = 0.0

    # f(p) = update_v_with_t!(p, t + 0.5dt)
    # g(p) = update_v_with_t!(p, t + dt)

    apply!(sys, p -> update_v!(p,t+0.5dt))
    # apply!(sys, f)
    apply!(sys, update_x!)
    create_cell_list!(sys)
    apply!(sys, reset!)
    apply!(sys, reset_L!)
    apply!(sys, find_L!)
    apply!(sys, update_A!)
    apply!(sys, relax_A!)
    apply!(sys, update_x!)
    create_cell_list!(sys)
    apply!(sys, reset!)
    apply!(sys, find_rho!)
    # apply!(sys, smoothing!, self=true)
    apply!(sys, find_T!)
    apply!(sys, find_f!)        
    apply!(sys, p -> update_v!(p,t+dt))
    # apply!(sys, g)
end

function test_benchmark_fast(sys)    
    
    function update_v_with_t!(p, t)
        (p -> update_v!(p, t))
    end

    t = 0.0

    f(p) = update_v_with_t!(p, t + 0.5dt)
    g(p) = update_v_with_t!(p, t + dt)

    # apply!(sys, p -> update_v!(p,t+0.5dt))
    apply!(sys, f)
    apply!(sys, update_x!)
    create_cell_list!(sys)
    apply!(sys, reset!)
    apply!(sys, reset_L!)
    apply!(sys, find_L!)
    apply!(sys, update_A!)
    apply!(sys, relax_A!)
    apply!(sys, update_x!)
    create_cell_list!(sys)
    apply!(sys, reset!)
    apply!(sys, find_rho!)
    # apply!(sys, smoothing!, self=true)
    apply!(sys, find_T!)
    apply!(sys, find_f!)        
    # apply!(sys, p -> update_v!(p,t+dt))
    apply!(sys, g)
end


end #module