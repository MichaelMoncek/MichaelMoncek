# TODO: add RK4
# TODO: find out the Schwarzchild radius and test if it still breaks the code
# TODO: optimize rest of the code - RK4
# TODO: make the disc more colorful
# The black hole will be centered at the origin.
# We will use the Schwarzschild metric to describe 
using LinearAlgebra
using Colors
using FileIO
using ImageIO
using Parameters
include("algebra.jl") 
# Parameters 
# image size 
const W = 600 # horizontal resolution 
const H = 600 # vertical resolution 
# camera
const camera_position = RealVector(1.0, 0.5, 4.0) # camera position in space
const fov = π/3 # field of view
const look_at = RealVector(0.0, 0.0, 0.0) # point the camera is looking at 
const world_up = RealVector(1.0, 0.0, 0.0) # world up vector 
# positions & physics 
const M = 0.1 #44.0 # mass of the black hole sphere and disc
const sphere_position = RealVector(0.0, 0.0, 0.0) # center of the sphere 
const sphere_radius = 3*M # radius of the sphere 
const second_sphere_position = RealVector(0.0, 0.0, -555.0)
const second_sphere_radius = 3.0 
const third_sphere_position = RealVector(1.0, 1.0, -553.0)
const third_sphere_radius = 0.9
const Rin = 0.1 # inner radius of the disc 
const Rout = 1.0 # outer radius of the disc 
# camera basics 
const F = normalize(look_at - camera_position) # where the camera is looking at 
const R = normalize(cross(F,world_up)) # horizontal axis of the camera
const U = normalize(cross(F,R)) # vertical axis of the camera -- note that U should be already normalized, we do it for convenience
const aspect_ratio = W / H # aspect ration 
const scale = tan(fov / 2) 


#Structs 
@with_kw mutable struct Ray
    position::Real4Vector
    velocity::Real4Vector
    pixel::Tuple{Int, Int} # pixel coordinates of the ray
    color::RGB{Float64} = RGB(0.0, 0.0, 0.0) # default color is black
    flag::Bool = true # flag to indicate if the ray is active or not
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

@inline function dir_to_sph_rates(dir::RealVector, r::Float64, θ::Float64, ϕ::Float64)::RealVector
    er,eθ,eϕ = sph_basis(r,θ,ϕ)
    dr = dir[1]*er[1] + dir[2]*er[2] + dir[3]*er[3]
    dθ = (dir[1]*eθ[1] + dir[2]*eθ[2] + dir[3]*eθ[3]) / r
    dϕ = (dir[1]*eϕ[1] + dir[2]*eϕ[2] + dir[3]*eϕ[3]) / (r*sin(θ))
    return RealVector(dr, dθ, dϕ)
end

function make_system!(rays::Vector{Ray}, W::Int, H::Int)::Vector{Ray}
    # This function creates array of rays with initial positions and velocities
    # that are moving away from the camera position. 
    #rays = SArray{Tuple{W*H}, undef, 1, W*H}
    #rays_index = 1
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
            #D = cartesian_to_spherical(D)
            #TODO: we need to initialize the ray velocity with right dt
            # r, theta, phi
            dr = D[1] 
            dtheta = D[2] 
            dphi = D[3]
            r = pos[1]
            theta = pos[2]
            fr = f(r)
            dt = sqrt( (dr^2)/(fr)^2) + (r^2*(dtheta^2 + sin(theta)^2 * dphi^2))/fr  
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
                check_for_intersection(rays[k])
            end
        end 

    end 
    # Output image
    image = Array{RGB{Float64}}(undef, H, W)
    
    @inbounds Threads.@threads for k in eachindex(rays)
        ray = rays[k]
        if ray.flag == true 
            image[ray.pixel[1], ray.pixel[2]] = RGB(0, 0, 0)
        else 
            image[ray.pixel[1], ray.pixel[2]] = ray.color 
        end 
    
    end
    return image
end

# This function does a single Runge-Kutta 4 step for the ray
function RK4_step(ray::Ray, h::Float64)
    # TODO: implement RK4 step 
    return nothing
end

# This function provides the right-hand side of the Schwarzschild equation 
function schwarzschild_rhs(ray::Ray)
    # TODO: implement the right-hand side of the Schwarzschild equation
    return nothing
end 

function move_ray(ray::Ray, h::Float64)
    # Euler iteration 
    r = ray.position[1]
    if r > 50.0 
        ray.flag == false
        #println("Ray is too far away, stopping it")
        return
    end 
    theta = ray.position[2]
    #phi = ray.position[3]
    #t = ray.position[4] # time coordinate     

    dr = ray.velocity[1] 
    dtheta = ray.velocity[2]
    dphi = ray.velocity[3]
    dt = ray.velocity[4] 

    # Update the velocities according to the Schwarzschild equation 
    # dr += h * (- 2 * M / (r^2 * f(r)) * dr^2 + r * f(r) * dt^2 + r * f(r) * sin(theta)^2 * dtheta^2 
    # - 0.5 * f(r) * 2 * M / r^2 * dt^2) # radial velocity update
    # dtheta += h * (- 2 / r * dtheta * dr + sin(theta) * cos(theta) * dtheta^2)
    # dphi += h * (- 2 / r * dphi * dr + 2 * cot(theta) * dphi * dtheta)
    # dt += h * (- 2 * M / (r^2 * f(r)) * dt * dr)
    fr = f(r)
    sin_theta = sin(theta)
    dr += h * ( (M/r^2) * (dr^2/fr - fr*dt^2) + r*fr*(dtheta^2 + sin_theta^2 * dphi^2) )
    dtheta += h * (-2/r * dr*dtheta + sin_theta*cos(theta)*dphi^2)
    dphi += h * (-2/r * dr*dphi - 2*cot(theta)*dtheta*dphi)
    dt += h * (-2*M/(r^2*fr) * dt*dr)


    # Update the position 
    # ray.position[1] += dr
    # ray.position[2] += dtheta
    # ray.position[3] += dphi
    # ray.position[4] += dt
    ray.position += h * Real4Vector(dr, dtheta, dphi, dt)
    ray.velocity = Real4Vector(dr, dtheta, dphi, dt)
    # Update the velocity
    # ray.velocity[1] = dr
    # ray.velocity[2] = dtheta
    # ray.velocity[3] = dphi
    # ray.velocity[4] = dt
end

# Convert spherical coordinates to cartesian coordinates for both 3D vectors
function spherical_to_cartesian(position::RealVector)::RealVector
    r = position[1]
    theta = position[2]
    phi = position[3]
    @fastmath sin_theta = sin(theta)
    @fastmath return RealVector(r*cos(phi)*sin_theta, r*cos(theta), r*sin(phi)*sin_theta)
end    

# Convert spherical coordinates to cartesian coordinates for 4D vectors
function spherical_to_cartesian(position::Real4Vector)::RealVector
    r = position[1]
    theta = position[2]
    phi = position[3]
    @fastmath sin_theta = sin(theta)
    @fastmath return RealVector(r*cos(phi)*sin_theta, r*cos(theta), r*sin(phi)*sin_theta)
end   

# Convert cartesian coordinates to spherical coordinates for both 3D and 4D vectors 
function cartesian_to_spherical(position::RealVector)::RealVector
    r = norm(position)
    theta = acos(position[2] / r)
    phi = atan(position[3], position[1])
    return RealVector(r, theta, phi)
end

function check_for_intersection(ray::Ray)
    # TODO: we need to convert the ray back to cartesian coordinates before checking for collisions  
    position = spherical_to_cartesian(ray.position)
    #R = norm(position - sphere_position)
    R2 = norm2(position - sphere_position)
    if R2 < sphere_radius^2
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
            ray.color = RGB(sqrt(rho) / Rout, sqrt(rho) / Rout, sqrt(rho) / Rout)
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
    image_curved = render(rays_curved, 100.0, 0.001)
    # Save the image
    save("sphere_test_curved_ray_casting.png", image_curved)
    println("Saved sphere_test.png")
end

main()