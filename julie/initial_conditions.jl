function crossproduct(a, b)
    return [a[2]*b[3] - a[3]*b[2], a[3]*b[1] - a[1]*b[3], a[1]*b[2] - a[2]*b[1]]
end

function initial_conditions(N, thickness, radius, T = Float64)
    sphere_min_radius = radius - thickness / 2
    sphere_max_radius = radius + thickness / 2
    # Generate random positions in a sphere of radius 
    radii = ((rand(N) .* (sphere_max_radius^3 - sphere_min_radius^3)) .+ sphere_min_radius^3).^(1/3)
    theta = acos.(1 .- 2*rand(N))
    phi = 2 * pi * rand(N)

    positions = hcat(
        radii .* sin.(theta) .* cos.(phi),
        radii .* sin.(theta) .* sin.(phi),
        radii .* cos.(theta)
    )
    up = [0, 0, 1]
    velocities = map(1:N) do i 
        direction = crossproduct(up, positions[i, :])
        direction ./= sqrt(sum(c->c^2, direction))
        return direction * (rand() * 2  + 5)
    end
    velocities = Matrix(transpose(hcat(velocities...)))
    velocities .*= 2

    masses = rand(N)

    # Set the inner particle to be stationary and very large
    masses[1] = 1000
    velocities[1, :] = [0, 0, 0]
    positions[1, :] = [0, 0, 0]

    com = sum(positions .* masses, dims=1) / sum(masses)
    positions .-= com

    com_velocity = sum(velocities .* masses, dims=1) / sum(masses)
    velocities .-= com_velocity

    return positions, velocities, masses
end