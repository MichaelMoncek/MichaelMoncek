using GLMakie
using LinearAlgebra

# Parameters
const L = 6.0           # half-size of floor
const n = 15            # number of grid points per axis
const rs = 1.0          # Schwarzschild radius
const M = rs / 2.0      # Mass of the black hole
const E = 1.0           # Total energy of the system 

# Schwarzschild metric 
# we are in the equatorial plane, so d theta = 0
# ds^2 = -f(r) dt^2 + f(r)^{-1}dr^2 + r^2 d phi^2
f(r) = 1 - 2 * M / r


# RHS of ODE: state = [t, r, phi, p_r]
function rhs!(dstate, state, params, λ)
    t, r, phi, p_r = state 
    #L = params.L 
    fr = f(r)
    dt_dλ = E / fr 
    dr_dλ = fr * p_r 
    dphi_λ = L / r^2
    dp_r_dλ = - M * E^2 / (r^2 * fr^2) - M * p_r^2 / r^2 + L^2 / r^3

    dstate[1] = dt_dλ
    dstate[2] = dr_dλ
    dstate[3] = dphi_λ
    dstate[4] = dp_r_dλ
    return nothing
end

# RK4 step
function rk4_step!(state, h, rhs!, params, λ)
    k1 = similar(state); rhs!(k1, state, params, λ)
    tmp = state .+ 0.5h .* k1
    k2 = similar(state); rhs!(k2, tmp, params, λ + 0.5h)
    tmp = state .+ 0.5h .* k2
    k3 = similar(state); rhs!(k3, tmp, params, λ + 0.5h)
    tmp = state .+ h .* k3
    k4 = similar(state); rhs!(k4, tmp, params, λ + h)
    state .+= (h/6.0) .* (k1 .+ 2k2 .+ 2k3 .+ k4)
    return nothing
end

# Integrate one ray until stop condition
function integrate_ray(r0, φ0, b; h=0.01, λmax=1e5, rmin=2M+1e-6, rexit=1000.0)
    L = b * E
    # initial Veff and dr/dλ
    fr0 = f(r0)
    V0 = fr0 * (L^2 / r0^2)
    if E^2 <= V0
        return (Float64[], Float64[])  # no valid incoming ray
    end
    drdλ0 = -sqrt(E^2 - V0)    # incoming
    p_r0 = drdλ0 / fr0

    state = [0.0, r0, φ0, p_r0]   # t, r, φ, p_r
    params = (; L=L)

    xs = Float64[]
    ys = Float64[]

    λ = 0.0
    while λ < λmax
        r = state[2]
        φ = state[3]
        # stop if fell in horizon or escaped far away
        if r <= rmin || r > rexit
            break
        end
        # record position in Cartesian (embedding z will be computed for plotting)
        push!(xs, r * cos(φ))
        push!(ys, r * sin(φ))

        rk4_step!(state, h, rhs!, params, λ)
        λ += h
    end
    return xs, ys
end

# Example: integrate a family of rays with different impact parameters
r_obs = 20.0            # start far away
φ_obs = 1.0
bs = [2.6, 3.0, 3.3, 3.6, 4.0, 5.0]  # try several impact parameters
ray_trajs = [ integrate_ray(r_obs, φ_obs, b; h=0.2, λmax=2e5, rexit=20.0) for b in bs ]

# Plotting 

# Create square grid in Cartesian coords
x = range(-L, L, length=n)
y = range(-L, L, length=n)

# Compute z values: funnel in the middle
z = [ begin
        r = sqrt(xi^2 + yi^2)
        if r >= rs
            2 * sqrt(rs * (r - rs))  # embedding formula
        else
            #0.0
            NaN                      # no surface inside event horizon
        end
     end for yi in y, xi in x ]

# Plot wireframe without axes, ticks, or borders
fig = Figure(resolution = (800, 800), backgroundcolor=:black)
ax = Axis3(fig[1, 1],
    aspect = :data,
    backgroundcolor = :black
)

wireframe!(ax, x, y, z, color = :white, linewidth = 1.0)


sphere_res = 20  # smaller = lower resolution

θ = range(0, π, length=sphere_res)
φ = range(0, 2π, length=sphere_res)

Θ = repeat(θ', sphere_res, 1)
Φ = repeat(φ, 1, sphere_res)

cx, cy, cz = 0.0, 0.0, z[1,1]  # center position

sphere_x = cx .+ rs .* sin.(Θ) .* cos.(Φ)
sphere_y = cy .+ rs .* sin.(Θ) .* sin.(Φ)
sphere_z = cz .+ rs .* cos.(Θ)

surface!(ax, sphere_x, sphere_y, sphere_z)

# overlay ray trajectories (transform to 3D by mapping z = embedding z(r))
for (i, (xs, ys)) in enumerate(ray_trajs)
    if length(xs) > 1
        zs = Float64[]
        for (xi, yi) in zip(xs, ys)
            rr = sqrt(xi^2 + yi^2)
            zz = rr >= rs ? 2 * sqrt(rs * (rr - rs)) : -rs  # inside hole put it low
            push!(zs, zz)
        end
        lines!(ax, xs, ys, zs, linewidth=2.0) # yellowish
    end
end

fig
