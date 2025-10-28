using DifferentialEquations, Plots

M = 1.0
r0 = 3.1
t0 = 0.0
ϕ0 = 0.0
E = 1.0

function geodesic!(du, u, p, λ)
    t, r, ϕ, p_r = u
    L, M = p
    f = 1 - 2M/r

    dt_dλ = E / f
    dr_dλ = f * p_r
    dϕ_dλ = L / r^2
    dp_r_dλ = - (M * E^2) / (r^2 * f^2) + (L^2) / r^3

    du[1] = dt_dλ
    du[2] = dr_dλ
    du[3] = dϕ_dλ
    du[4] = dp_r_dλ
    # println("geodesic called")
end

function condition(u, t, integrator)
    r = u[2]
    return r - 2M - 1e-6
end

function affect!(integrator)
    terminate!(integrator)
    println("terminating")
end

cb = ContinuousCallback(condition, affect!)

# Array of L values to launch rays
L_values = range(2.8, 4.0, length=30)

plot(aspect_ratio=:equal, xlabel="x", ylabel="y", legend=false, title="Photon trajectories near black hole")

scatter!([0], [0], color=:black, label="Black Hole")

for L in L_values
    f0 = 1 - 2M/r0
    dϕ_dλ = L / r0^2
    dt_dλ = E / f0

    val = f0 * dt_dλ^2 - r0^2 * dϕ_dλ^2 * f0
    if val < 0
        println("Skipping L=$L due to negative sqrt")
        continue
    end

    for sign in (+1, -1)
        dr_dλ = sign * sqrt(val)
        p_r0 = dr_dλ / f0

        println("Launching ray with L=$L, p_r0=$p_r0")

        u0 = [t0, r0, ϕ0, p_r0]
        p = [L, M]

        prob = ODEProblem(geodesic!, u0, (0.0, 200.0), p)
        sol = solve(prob, Tsit5(), callback=cb, abstol=1e-9, reltol=1e-9)

        println("Solution length: ", length(sol.t))

        if length(sol.t) > 1
            r_vals = sol[2,:]
            phi_vals = sol[3,:]
            x_vals = r_vals .* cos.(phi_vals)
            y_vals = r_vals .* sin.(phi_vals)
            plot!(x_vals, y_vals, lw=1, alpha=0.6)
        else
            println("Short trajectory for L=$L, skipping plot")
        end
    end
end

plot!(xlims=(-10,10), ylims=(-10,10), aspect_ratio=:equal, xlabel="x", ylabel="y", legend=false, title="Photon trajectories near black hole")
#scatter!([0], [0], color=:black, label="Black Hole")

#display(plot())



