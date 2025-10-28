# TODO: add RK4
# TODO: find out the Schwarzchild radius and test if it still breaks the code
# TODO: optimize rest of the code - RK4
# TODO: make the disc more colorful
using LinearAlgebra
using Colors
using FileIO
using ImageIO
using Parameters
include("algebra.jl") 
# Parameters 
# image size 
const W = 50 # horizontal resolution
const H = 50 # vertical resolution 
# camera
const camera_position = RealVector(1.0, 0.3, 4.0) # camera position in space
# field of view
const fov = π/3 
const look_at = RealVector(0.0, 0.0, 0.0) # point the camera is looking at 
const world_up = RealVector(1.0, 0.0, 0.0) # world up vector 
# positions & physics 
const M = 0.1 #44.0 # mass of the black hole sphere and disc
const sphere_position = RealVector(0.0, 0.0, 0.0) # center of the sphere 
const sphere_radius = 2*M # radius of the sphere 
const second_sphere_position = RealVector(0.0, 0.0, -555.0)
const second_sphere_radius = 3.0 
const third_sphere_position = RealVector(1.0, 1.0, -553.0)
const third_sphere_radius = 0.9
const Rin = 0.1 # inner radius of the disc 
const Rout = 2.0 # outer radius of the disc 
# camera basics 
const F = normalize(look_at - camera_position) # where the camera is looking at 
const R = normalize(cross(F,world_up)) # horizontal axis of the camera
const U = normalize(cross(F,R)) # vertical axis of the camera -- note that U should be already normalized, we do it for convenience
const aspect_ratio = W / H # aspect ration 
const scale = tan(fov / 2) 
const EPS_TH = 1e-8


#Structs 
@with_kw mutable struct Ray
    position::Real4Vector
    velocity::Real4Vector
    pixel::Tuple{Int, Int} # pixel coordinates of the ray
    color::RGB{Float64} = RGB(0.0, 0.0, 0.0) # default color is black
    flag::Bool = true # flag to indicate if the ray is active or not
    #isActive::Bool = true # flag to indicate if the ray is active or not
end 

#Functions 
@inline function map_pixel(x::Int, N::Int)::Float64
    return ((x - 0.5) / N - 0.5) * 2.0
end

@inline function f(r::Float64)::Float64
    return 1 - 2 * M / r 
end 

@inline function to4Vector(x::RealVector, a::Float64)::Real4Vector
    # Convert a RealVector to a Real4Vector by appending zero
    return Real4Vector(x[1], x[2], x[3], a)
end 

@inline function return_index(i::Int, j::Int, W::Int, H::Int)::Int 
    # This function return the index of the ray in the rays array 
    return (j - 1) * W + i 
end

@inline function sph_basis(r::Float64, theta::Float64, phi::Float64)
    er  = RealVector(cos(phi)*sin(theta),      cos(theta), sin(phi)*sin(theta))
    etheta  = RealVector(cos(phi)*cos(theta), -sin(theta), cos(theta)*sin(phi))
    ephi  = RealVector(-sin(phi),                     0.0,            cos(phi))
    return er, etheta, ephi
end

@inline function dir_to_sph_rates(dir::RealVector, r::Float64, theta::Float64, phi::Float64)::RealVector
    er,etheta,ephi = sph_basis(r,theta,phi)
    dr = dir[1]*er[1] + dir[2]*er[2] + dir[3]*er[3]
    dtheta = (dir[1]*etheta[1] + dir[2]*etheta[2] + dir[3]*etheta[3]) / r
    dphi = (dir[1]*ephi[1] + dir[2]*ephi[2] + dir[3]*ephi[3]) / (r*sin(theta))
    return RealVector(dr, dtheta, dphi)
end

@inline function sanitize_angles!(x::Real4Vector, v::Real4Vector)
    r, θ, ϕ, t = x
    dr, dθ, dϕ, dt = v

    # wrap φ to (-π, π]
    ϕ = mod(ϕ + π, 2π) - π

    # reflect θ at poles to stay within [ε, π-ε]
    if θ < EPS_TH
        θ = 2EPS_TH - θ
        dθ = -dθ
    elseif θ > π - EPS_TH
        θ = 2(π - EPS_TH) - θ
        dθ = -dθ
    end

    # cap absurd angular rates to avoid blow-ups in one step
    if !isfinite(dϕ) || abs(dϕ) > 1e6
        dϕ = sign(dϕ)*1e6
    end
    if !isfinite(dθ) || abs(dθ) > 1e6
        dθ = sign(dθ)*1e6
    end

    return Real4Vector(r, θ, ϕ, t), Real4Vector(dr, dθ, dϕ, dt)
end

@inline function mod_angles!(x::Real4Vector)::Real4Vector
    x2 = x[2] % (2*pi)
    x3 = x[3] % (2*pi)
    return Real4Vector(x[1], x2, x3, x[4]) 
end




function make_system!(rays::Vector{Ray}, W::Int, H::Int)::Vector{Ray}
    # This function creates array of rays with initial positions and velocities
    # that are moving away from the camera position. 
    # using Threads in this place is not a great improvement, however if one should run 
    # interactive simulation where make_system is called multiple times, this will save a
    # lot of time 
    @inbounds Threads.@threads for j in 1:H
        @inbounds @simd for i in 1:W 
            u = map_pixel(i, W)
            v = map_pixel(j, H)

            # apply FOV scaling 
            px = u * aspect_ratio * scale
            py = -v * scale # minus sign so that the y-axis points up

            # ray direction 
            cam_buffer = cartesian_to_spherical(camera_position)
            pos = to4Vector(cam_buffer, 0.0) 

            D = normalize(F + px*R + py*U)
            D = dir_to_sph_rates(D, cam_buffer[1], cam_buffer[2], cam_buffer[3])
            dr = D[1] 
            dtheta = D[2] 
            dphi = D[3]
            r = pos[1]
            theta = pos[2]
            fr = f(r)
            dt = sqrt( (dr^2)/(fr)^2) + (r^2*(dtheta^2 + sin(theta)^2 * dphi^2))/fr    #TODO: shouldn't fr be outside?
            D_spherical = to4Vector(D, dt) 
             
            pix = (i, j)
            ray = Ray(position=pos, velocity=D_spherical, pixel=pix)
            rays[return_index(i, j, W, H)] = ray
            # rays[rays_index] = ray
            # rays_index += 1 # this will cause a racing condition if we use multiple  
            # # one should use pixel to determine the index
        end 
    end 
    #println("Created system of rays with curved coordinates")
    return rays 
end

# This function moves the rays and renders the image
function render(rays::Array, t::Float64, dt::Float64 = 0.01)
   for t in 0:dt:t
        # For each ray in rays array move the ray
        # check for intersection with the sphere and the disc, if there is a collision, remove the ray from the array
        # (it needs no longer to be moved so we would be wasting time)
        Threads.@threads for k in eachindex(rays)
            # Move each ray only if it is still active
            @inbounds if rays[k].flag == true  
                move_ray(rays[k], dt)
                #RK4_step!(rays[k], dt)
                check_for_intersection(rays[k])
                # if t == 4.123
                #     println("ray n$k position is: ", rays[k].position)
                # end
            end
        end 
        #println(t)
    end 
    # Output image
    image = Array{RGB{Float64}}(undef, H, W)
    
    @inbounds Threads.@threads for k in eachindex(rays)
        ray = rays[k]
        image[ray.pixel[1], ray.pixel[2]] = ray.color
    end
    return image
end

# This function does a single Runge-Kutta 4 step for the ray
function RK4_step!(ray::Ray, h::Float64)
    # TODO: implement RK4 step 
    #h = h * clamp((ray.position[1] - 2M) / (0.5), 0.05, 1.0)
    x, v = ray.position, ray.velocity
    # x = mod_angles!(x)
    v = mod_angles!(v)

    k1x = v
    k1v = schwarzschild_rhs(x, v)
    # k1x = mod_angles!(k1x)
    k1v = mod_angles!(k1v)
    
    k2x = v + 0.5h * k1v
    k2v = schwarzschild_rhs(x + 0.5h * k1x, v + 0.5h * k1v)
    # k2x = mod_angles!(k2x)
    k2v = mod_angles!(k2v)

    k3x = v + 0.5h * k2v
    k3v = schwarzschild_rhs(x + 0.5h * k2x, v + 0.5h * k2v)
    # k3x = mod_angles!(k3x)
    k3v = mod_angles!(k3v)

    k4x = v + h * k3v
    k4v = schwarzschild_rhs(x + h * k3x, v + h * k3v)
    # k4x = mod_angles!(k4x)
    k4v = mod_angles!(k4v)


 
    # update position and velocity
    position = ray.position + h/6 * (k1x + 2*k2x + 2*k3x + k4x)
    velocity = ray.velocity + h/6 * (k1v + 2*k2v + 2*k3v + k4v)
    #ray.position, ray.velocity = sanitize_angles!(position, velocity)
    ray.position = mod_angles!(position)
    ray.velocity = mod_angles!(velocity) 
end

# This function provides the right-hand side of the Schwarzschild equation 
function schwarzschild_rhs(position::Real4Vector, velocity::Real4Vector)::Real4Vector
    r = position[1]
    theta = position[2]
    dr = velocity[1]
    dtheta = velocity[2]
    dphi = velocity[3]
    dt = velocity[4]
    sin_theta = sin(theta)
    fr = f(r)

    ddr = ((M/r^2) * (dr^2/fr - fr*dt^2) + r*fr*(dtheta^2 + sin_theta^2 * dphi^2))
    ddtheta = (-2/r * dr*dtheta + sin_theta*cos(theta)*dphi^2)
    ddphi = (-2/r * dr*dphi - 2*cot(theta)*dtheta*dphi)
    ddt = (-2*M/(r^2*fr) * dt*dr)   
    return Real4Vector(ddr, ddtheta, ddphi, ddt) 
end 

function move_ray(ray::Ray, h::Float64)
    # Euler iteration 
    r = ray.position[1]
    if r > 50
        ray.flag = false
        #println("Ray is too far away, stopping it")
        return
    end 
   
    ray.velocity += h * schwarzschild_rhs(ray.position, ray.velocity)
    ray.position += h * ray.velocity
end

# Convert spherical coordinates to cartesian coordinates for both 3D vectors
function spherical_to_cartesian(position::RealVector)::RealVector
    r = position[1]
    theta = position[2]
    phi = position[3]
    sin_theta = sin(theta)
    return RealVector(r*cos(phi)*sin_theta, r*cos(theta), r*sin(phi)*sin_theta)
end    

# Convert spherical coordinates to cartesian coordinates for 4D vectors
function spherical_to_cartesian(position::Real4Vector)::RealVector
    r = position[1]
    theta = position[2]
    phi = position[3]
    sin_theta = sin(theta)
    return RealVector(r*cos(phi)*sin_theta, r*cos(theta), r*sin(phi)*sin_theta)
end   

# Convert cartesian coordinates to spherical coordinates for both 3D and 4D vectors 
function cartesian_to_spherical(position::RealVector)::RealVector
    r = norm(position)
    theta = acos(position[2] / r)
    phi = atan(position[3], position[1])
    return RealVector(r, theta, phi)
end

# Color the disc based on radius and agnle
function disc_color(r::Float64, phi::Float64, Rin::Float64, Rout::Float64)::RGB
    # normalize radius to [0,1]
    t = clamp((Rout - r) / (Rout - Rin), 0, 1)

    #--- base gradient: orange → yellow → white
    base = (
        (1.0, 0.5, 0.1),   # orange at outer edge
        (1.0, 0.9, 0.2),   # yellow
        (1.0, 1.0, 1.0)    # white at inner edge
    )
   
    # simple 2-step linear blend
    if t < 0.5
        f = 2t
        color = (
            base[1][1]*(1-f) + base[2][1]*f,
            base[1][2]*(1-f) + base[2][2]*f,
            base[1][3]*(1-f) + base[2][3]*f
        )
    else
        f = 2(t-0.5)
        color = (
            base[2][1]*(1-f) + base[3][1]*f,
            base[2][2]*(1-f) + base[3][2]*f,
            base[2][3]*(1-f) + base[3][3]*f
        )
    end

    # --- radial noise: ring pattern
    ring_pattern = 0.85 + 0.25*sin(40r + 5rand())
    brightness = t^1.3 * ring_pattern

    # --- azimuthal streaks (adds turbulence)
    streaks = 1.0 + 0.1*sin(50phi + 7rand())
    brightness *= streaks

    # clamp and apply brightness
    color = RGB(
        clamp(color[1]*brightness, 0, 1),
        clamp(color[2]*brightness, 0, 1),
        clamp(color[3]*brightness, 0, 1)
    )

    return color
end


function check_for_intersection(ray::Ray)
    # TODO: we need to convert the ray back to cartesian coordinates before checking for collisions  
    position = spherical_to_cartesian(ray.position)
    #R = norm(position - sphere_position)
    R2 = norm2(position - sphere_position)
    if R2 < sphere_radius^2 + 1e-3      # stop the ray tiny bit outside the black hole for stability
        # ray hit the sphere, we can stop it
        #ray.color = RGB(1, 1, 1)
        ray.color = RGB(0, 0, 0)
        ray.flag = false
        # println("spherical hit")
        # println(position[1])
        # println(ray.position)
        return 
    end
    R_second2 = norm2(position - second_sphere_position )
    if R_second2 < second_sphere_radius^2
        ray.color = RGB(1, 0, 0) 
        ray.flag = false
        # println("spherical hit")
        # println(position[1])
        # println(ray.position)
        return 
    end 
    R_third2 = norm2(position - third_sphere_position)
    if R_third2 < third_sphere_radius^2
        ray.color = RGB(0, 1, 0) 
        # println("spherical hit")
        # println(position[1])
        # println(ray.position)
        # ray.flag = false
        return
    end
    if abs(position[2]) < 5e-3  
        rho = position[1]^2 + position[3]^2 
        if Rin^2 <= rho <= Rout^2
            #ray.color = RGB(sqrt(rho) / Rout, sqrt(rho) / Rout, sqrt(rho) / Rout)
            ray.color = disc_color(ray.position[1], ray.position[2], Rin, Rout)
            ray.flag = false
            # println("disc hit")
            # println(position[1])
            # println(ray.position)
            return
        end          
    end
end



function main()
    # rays = make_system(W, H)
    rays = Vector{Ray}(undef, W*H)
    rays_curved = make_system!(rays, W, H)
    # Move the rays
    image_curved = render(rays_curved, 100.0, 0.01)
    # Save the image
    save("sphere_test_curved_ray_casting2.png", image_curved)
    println("Saved sphere_test.png")
end

