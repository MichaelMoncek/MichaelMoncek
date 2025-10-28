#=

# 8: Vertical heat convection

```@raw html
	<img src='../assets/fixpa.png' alt='missing' width="50%" height="50%" /><br>
```
Simulation of heat convection
=#

#=
CHANGES TO STOP PARTICLES ESCAPING:
-increase E_wall
-increase wall width
seems to have worked :)
TODO
- change particle struct to include heat flux
- check if heat flux addition is correct in bc function
- in main should it be Q_COOLER = new_Q_COOLER or Q_COOLER += new_Q_COOLER?
- calculate Nusselt number
=#

#=
USEFULL COMMANDS:
pull results:
scp -r -C hart@r3d3.karlin.mff.cuni.cz:~/heatHe/heatHe/ ~/vysledky/
push code:
scp -r -C ~/bakalarka/rayleigh-benard/heat_He.jl hart@r3d3.karlin.mff.cuni.cz:~/heatHe/
=#

module heat_He

using Printf
using SmoothedParticles
using Parameters
using Plots
using DataFrames # to store the csv file
using CSV# to store the csv file
include("utils/FixPA.jl")
include("utils/entropy.jl")
include("utils/ICR.jl")
using .FixPA
using .entropy
using LaTeXStrings #for better legends
using Random, Distributions
using LsqFit


#using ReadVTK  #not implemented
#using VTKDataIO

#=
Declare constant parameters
=#

##physical -- Helium at 5K, pressure = 1 bar, delta T = 50mK data from HEPAK
const dr = 1.0e-2/2           # average particle distance (decrease to make finer simulation)
const h = 5.0*dr            # size of kernel support
const g = -9.8*VECY         # gravitational acceleration
const mu = 1.39e-6          # dynamic viscosity of He -- from HEPAK
const gamma = 2.132         # cp/cv


const cv = 3159.0       # specific heat capacity
const p0 = 100000       # reference pressure in Pa
const rho0 = 11.78      # density in kg/m^3
const alpha = 1.631     # thermal expansion coefficient -- from HEPAK
const kappa = 1.29e-7   # thermal diffusivity -- from HEPAK
const c0 = 119.7        # speed of sound in m/s from HEPAK
const m = rho0*dr^2     #particle mass
const kB = 1.380649e-23
const T0 = 5.0          # initial temperature in K
#const T0 = c0^2/(gamma*(gamma-1.0)*cv) # from stiffened gas model
#const c0 = sqrt(p0*gamma/rho0)  # from stiffened gas model ... or is it?
@show T0

const m0 = rho0*dr*dr
const S0 = m0
@show rho0
@show m0
@show S0
@show c0
const Tdown = 5.05          #temperature at the bottom
const Tup= 5.0              #temperature at the top
const T_diff = Tdown - Tup  # for Ra
@show Tdown
@show Tup
const lambda = 0.01022      #heat exchange coefficient at the boundary
const bc_width = h
const lambda_F = 0.01022    #heat conductivity (times temperature squared)
const Q_tol = 0.0001        # tolerance for measuring heat fluxes
const nu = mu/rho0      #kinematic viscosity


##geometrical
const box_height = 0.5
const box_width = 0.5
const wall_width = 5.0*dr # 2.5*dr

##artificial
const dr_wall = 0.95*dr
const E_wall = 10*norm(g)*100
const eps = 1e-6

##temporal
const dt = 0.01*h/c0
@show dt
const t_end = 1.5
@show t_end
const dt_frame = t_end/500
@show dt_frame

##particle types
const FLUID = 0.
const WALL = 1.
const EMPTY = 2.

const Ra = (9.81*alpha*(Tdown - Tup)*box_height^3)/(nu*kappa)
@show Ra
const Pr = nu/kappa
@show Pr

folder_name = "heatHe"

mutable struct Particle <: AbstractParticle
	x::RealVector		# position
    m::Float64			# mass
    S::Float64 			# entropy
    v::RealVector 		# velocity
    a::RealVector 		# acceleration
    rho::Float64 		# density
    rho0::Float64  		# reference density
    s::Float64  		# entropy density
    P::Float64 			# pressure
    T::Float64 			# temperature
    q::RealVector 		# heat
	type::Float64 		# particle type
    QC::Float64         # heat received from cooler
    QH::Float64         # heat received from heater
	Particle(x::RealVector, type::Float64) = begin
		return new(x, m0, S0, VEC0, VEC0, 0.0, 0.0, 0.0, 0.0, 0.0, VEC0, type, 0.0, 0.0)
	end
end

#=
Define geometry and make particles
=#

function make_system()
	grid = Grid(dr, :square)
	box = Rectangle(0., 0., box_width, box_height)
	wall = BoundaryLayer(box, grid, wall_width)
	sys = ParticleSystem(Particle, box + wall, h)


	generate_particles!(sys, grid, box, x -> Particle(x, FLUID))
	generate_particles!(sys, grid, wall, x -> Particle(x, WALL))

	return sys
end

#=
Define particle interactions
=#

@inbounds function internal_force!(p::Particle, q::Particle, r::Float64)
	if p.type == FLUID && q.type == FLUID
		ker = q.m*rDwendland2(h,r)
        x_pq = p.x - q.x
        p.a += -ker*(p.P/p.rho^2 + q.P/q.rho^2)*(p.x - q.x) # pressure
    	p.a += 8.0*ker*mu/(p.rho*q.rho)*dot(p.v-q.v, x_pq)/(r*r + 0.01*h*h)*x_pq # viscosity
	elseif p.type == FLUID && q.type == WALL && r < dr_wall
		s2 = (dr_wall^2 + eps^2)/(r^2 + eps^2)
		p.a += -E_wall/(r^2 + eps^2)*(s2 - s2^2)*(p.x - q.x)
	end
end

function reset_a!(p::Particle)
    p.a = zero(RealVector)
end

function reset_rho!(p::Particle)
    p.rho = 0.0
end

function move!(p::Particle)
	if p.type == FLUID
		p.a = VEC0
		p.x += dt*p.v
		#reset rho, s and a
		p.rho = 0.
		p.s = 0.
        p.q = VEC0
		p.a = VEC0
	end
end

function accelerate!(p::Particle)
	if p.type == FLUID
		p.v = p.v + 0.5*dt*(p.a + g)
	end
end

@inbounds function find_rho!(p::Particle, q::Particle, r::Float64)
	if p.type == FLUID && q.type == FLUID
        p.rho += q.m*wendland2(h,r)
        #p.s   += q.S*wendland2(h,r)
    end
end

@inbounds function apply_rho0!(p::Particle)
	if p.type == FLUID
		p.rho += p.rho0
	end
end

@inbounds function find_s!(p::Particle)
	if p.type == FLUID
        p.s   = p.S*p.rho/p.m
    end
end

@inbounds function find_rho0!(p::Particle, q::Particle, r::Float64)
    if p.type == FLUID && q.type == FLUID
		p.rho0 += m*wendland2(h,r)
	end
end

@inbounds function set_rho0!(p::Particle)
    if p.type == FLUID
		p.rho0 = rho0*c0^2/(gamma*(gamma-1.0)*cv*T0)
	end
end

function eint(rho::Float64, s::Float64)::Float64
	return rho*c0^2/(gamma*(gamma-1.0))*(rho/rho0)^(gamma-1.0)*exp(s/(cv*rho))+rho0*c0^2/gamma - p0
end


function find_P!(p::Particle)
	if p.type == FLUID
    	p.T = c0^2/(gamma*(gamma-1.0))*(p.rho/rho0)^(gamma-1.0)*exp(p.s/(p.rho*cv))/cv
		p.P = (gamma-1.0)*eint(p.rho, p.s)- (rho0*c0^2-gamma*p0)
	end
end

function LJ_potential(p::Particle, q::Particle, r::Float64)::Float64
	if q.type == WALL && p.type == FLUID && r < dr_wall
		s2 = (dr_wall^2 + eps^2)/(r^2 + eps^2)
		return m*E_wall*(0.25*s2^2 - 0.5*s2 + 0.25)
	else
		return 0.0
	end
end

function energy_kinetic(sys::ParticleSystem)::Float64
	return sum(p -> 0.5*m*dot(p.v, p.v), sys.particles)
end

function energy(sys::ParticleSystem)
	(E_kin, E_int, E_gra, E_wal, E_tot) = (0., 0., 0., 0., 0.)
	for p in sys.particles
		if p.type == FLUID
			E_kin += 0.5*m*dot(p.v, p.v)
			E_int += eint(p.rho, p.s)/p.rho*m
			#E_int +=  0.5*m*c^2*(p.rho - p.rho0)^2/rho0^2
			E_gra += -m*dot(g, p.x)
			E_wal += SmoothedParticles.sum(sys, LJ_potential, p)
		end
	end
	E_tot = E_kin +E_int + E_wal + E_gra
	return (E_tot, E_kin, E_int, E_gra, E_wal)
end

function bc!(sys::ParticleSystem)
	for p in sys.particles
		if p.type == FLUID && p.x[2] < bc_width # bottom
			p.S += p.m * lambda * cv * (Tdown-p.T) * dt
            p.QH += p.m * lambda * cv * (Tdown-p.T) * dt
		end
		if p.type == FLUID && p.x[2] > box_height - bc_width #top
			p.S += p.m * lambda * cv * (Tup -p.T) * dt
            p.QC += p.m * lambda * cv * (Tup -p.T) * dt
		end
	end
end

function find_heat!(p::Particle, q::Particle, r::Float64)
	if p.type == FLUID && q.type == FLUID
		ker = rDwendland2(h,r)
		x_pq = p.x - q.x
        p.q += lambda_F * q.m/q.rho*p.T*(1.0-p.T/q.T)*ker*x_pq # diffusion
    end
end

function entropy_production!(p::Particle, q::Particle, r::Float64)
	if p.type == FLUID && q.type == FLUID
		ker = rDwendland2(h,r)
		x_pq = p.x - q.x
		v_pq = p.v - q.v
    	p.S += - 4.0*p.m*q.m*ker*mu/(p.T*p.rho*q.rho)*dot(v_pq, x_pq)^2/(r*r + 0.01*h*h)*dt #viscous
        p.S += p.m*q.m/(p.rho*p.T*q.rho)*dot(p.q+q.q,x_pq)*ker*dt  # Fourier
	end
end

function measure_heat_fluxes(sys::ParticleSystem, dt::Float64)
    q_COOLER = 0.0
    q_HEATER = 0.0
    for p in sys.particles
        if p.type == FLUID
            q_COOLER += p.QC
            q_HEATER += p.QH
            # reset the cummulative heat fluxes
            p.QC = 0.0
            p.QH = 0.0
        end
    end
    return q_COOLER/dt, q_HEATER/dt
end


function verlet_step!(sys::ParticleSystem)
    apply!(sys, accelerate!)
    apply!(sys, move!)
    create_cell_list!(sys)
    #apply!(sys, reset_rho!)
    #apply!(sys, find_rho!, self = true)
    #apply!(sys, find_pressure!)
    apply!(sys, reset_a!)
    apply!(sys, find_rho!, self=true)
	apply!(sys, apply_rho0!)
    apply!(sys, find_s!)
    apply!(sys, find_P!)
	apply!(sys, find_heat!)
	apply!(sys, entropy_production!)
    apply!(sys, internal_force!)
    apply!(sys, accelerate!)
end


function save_results!(out::SmoothedParticles.DataStorage, sys::ParticleSystem, k::Int64, E0::Float64, last_save_time::Ref{Float64})
    current_time = time() # Get the current time in seconds since the epoch
    if (k % Int64(round(dt_frame/dt)) == 0)
        save_frame!(out, sys, :v, :a, :type, :P, :s, :T, :rho)
        last_save_time[] = current_time # Update the last save time
    end

    # Check if more than 30 minutes (1800 seconds) have passed since the last save
    if current_time - last_save_time[] > 1800
        error("No frame has been saved in the last 30 minutes. Terminating simulation.")
    end
end

#=
Put everything into a time loop
=#

function main(;heating = true) #if heating=true, the bottom edge is heated to Tdown and the upper edge cooled to Tup
    sys = make_system()
    out = new_pvd_file(folder_name)
    last_save_time = Ref(time()) # Initialize the last save time
    # Initialization
    create_cell_list!(sys)
    apply!(sys, set_rho0!)
    apply!(sys, find_rho!, self = true)
    apply!(sys, apply_rho0!)
    apply!(sys, find_s!)
    apply!(sys, find_P!)
    apply!(sys, find_heat!)
    apply!(sys, internal_force!)

    N_of_particles = length(sys.particles)
    @show N_of_particles
    @show m

    step_final = Int64(round(t_end/dt))
    times = Float64[]       # Time instants
    E0 = energy(sys)[1]
    initial_T = average_T(sys)
    @show initial_T
    Ts = Float64[]          # Entropy values
    Ekin = Float64[]        # Kinetic energy values
    Ewall = Float64[]       # Wall energy values
    Eint = Float64[]        # Internal energy values
    Eg = Float64[]          # Gravitational energy values
    Etot = Float64[]        # Total energy values
    Qs_COOLER = Float64[]   # Heat fluxes
    Qs_HEATER = Float64[]   # Heat fluxes
    Q_COOLER = 0.0
    Q_HEATER = 0.0
    t_averaging = 0.0
    Nus = Float64[]         # Nusselt number

    @show length(sys.particles)
    for k = 0:step_final
        verlet_step!(sys)
        if heating
            bc!(sys)
        end
        save_results!(out, sys, k, E0, last_save_time)
        if k % round(step_final/100) == 0 # Store a number of entropy values
            @printf("t = %.6e\n", k*dt)
            push!(times, k*dt)
            # Measure heat fluxes
            t_averaging += dt
            if heating
                new_Q_COOLER, new_Q_HEATER = measure_heat_fluxes(sys, t_averaging)
            end
            Q_COOLER = new_Q_COOLER
            Q_HEATER = new_Q_HEATER
            t_averaging = 0.0
            push!(Qs_COOLER, Q_COOLER)
            push!(Qs_HEATER, Q_HEATER)
            @show Q_COOLER
            @show Q_HEATER
            Nu = box_height*(Q_HEATER-Q_COOLER)/(lambda*(Tdown-Tup))
            @show Nu
            push!(Nus, Nu)
            # Energy
            (E_tot, E_kin, E_int, E_g, E_wal) = energy(sys)
            @show E_tot
            push!(Etot, E_tot)
            @show E_kin
            push!(Ekin, E_kin)
            @show E_int
            push!(Eint, E_int)
            @show E_g
            push!(Eg, E_g)
            @show E_wal
            push!(Ewall, E_wal)
            E_err = E_tot - E0
            @show E_err

            T = average_T(sys)
            push!(Ts, T)
            @show T
            println("# of part. = ", length(sys.particles))
            println()
            if length(sys.particles) != N_of_particles
                error("The number of particles has changed during the simulation.")
            end
        end
    end

    # Plotting the energies in time
    p = plot(times, Etot, label = "E_tot", legend=:bottomright)
    savefig(p, folder_name*"/Etot.pdf")
    p = plot(times, Ekin, label = "E_kin", legend=:bottomright)
    savefig(p, folder_name*"/Ekin.pdf")
    p = plot(times, Eint, label = "E_int", legend=:bottomright)
    savefig(p, folder_name*"/Eint.pdf")
    p = plot(times, Eg, label = "E_g", legend=:bottomright)
    savefig(p, folder_name*"/Eg.pdf")
    p = plot(times, Ewall, label = "E_wall", legend=:bottomright)
    savefig(p, folder_name*"/Ewall.pdf")
    p = plot(times, Ts, label = "T", legend=:bottomright)
    savefig(p, folder_name*"/T.pdf")
    p = plot(times, Nus, label = "Nu", legend=:bottomright)
    savefig(p, folder_name*"/Nu.pdf")

    df = DataFrame(
        time_steps = times, 
        E_total = Etot, 
        E_kinetic = Ekin, 
        E_internal = Eint, 
        E_walls = Ewall, 
        E_g = Eg, 
        temperature = Ts,
        Q_COOLER = Qs_COOLER,
        Q_HEATER = Qs_HEATER,
        Nusselt = Nus)
    CSV.write(folder_name*"/results.csv", df)

    final_T = average_T(sys)
    @show initial_T
    @show final_T

    save_pvd_file(out)
end

function plot_energy(energy_file::String)
    df = DataFrame(CSV.File(energy_file))
    times = df[:, "time_steps"]
    e_pot = df[:, "E_graviational"]
	Delta_e_pot = e_pot[1]-e_pot[end]
	print("Delta e pot = ", Delta_e_pot)
    e_tot = df[:, "E_total"]
	e_tot0 = e_tot[1]
	e_tot = (e_tot .- e_tot0)./Delta_e_pot
    p = plot(times, e_tot, legend=:topright, label=L"\frac{E_{tot}-E_{tot}(0)}{E_g(end)-E_g(0)}")
	savefig(p, "./energy_tot_scaled.pdf")
end

function average_T(sys::ParticleSystem)::Float64
    T = 0.0
    n = 0
    for p in sys.particles
        if p.type == FLUID
            T += p.T
            n += 1
        end
    end
    return T/n
end

end ## module

