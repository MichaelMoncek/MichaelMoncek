# TODO: 
# The black hole will be centered at the origin.
# We will use the Schwarzschild metric to describe 
using LinearAlgebra
using Colors
using FileIO
using ImageIO
using Parameters
# Parameters 
# image size 
const W = 100 # horizontal resolution 
const H = 100 # vertical resolution 
# camera
const camera_position = [0.0, 1.0, -3.0] # camera position in space
const fov = π/3 # field of view
const look_at = [0.0, 0.0, 0.0] # point the camera is looking at 
const world_up = [0.0, 1.0, 0.0] # world up vector 
# sphere and disc
const sphere_position = [0.0, 0.0, 0.0] # center of the sphere 
const sphere_radius = 0.5 # radius of the sphere 
const second_sphere_position = [0.0, 0.0, 1.0]
const second_sphere_radius = 0.7 
const third_sphere_position = [1.0, -1.0, 1.0]
const third_sphere_radius = 0.3
const Rin = 0.6 # inner radius of the disc 
const Rout = 1.0 # outer radius of the disc 
# camera basics 
const F = normalize(look_at - camera_position) # where the camera is looking at 
const R = normalize(cross(F, world_up)) # horizontal axis of the camera
const U = normalize(cross(R, F)) # vertical axis of the camera -- note that U should be already normalized, we do it for convenience
const aspect_ratio = W / H # aspect ration 
const scale = tan(fov / 2) 

#Structs 
@with_kw mutable struct Ray
    position::Vector{Float64} 
    velocity::Vector{Float64}
    pixel::Tuple{Int, Int} # pixel coordinates of the ray
    color::RGB{Float64} = RGB(0.0, 0.0, 0.0) # default color is black
    flag::Int8 = 0 # flag to indicate if the ray is active or not
end 

@inline function map_pixel(x::Int, N::Int)::Float64
    return ((x - 0.5) / N - 0.5) * 2.0
end


# # Stores the closest intersection distance for each pixel 
# closest_t = fill(Inf, W, H)

# for j in 1:H
#     for i in 1:W 
#         # normalize pixel coordinates
#         u = map_pixel(i, W)
#         v = map_pixel(j, H)

#         # apply FOV scaling 
#         px = u * aspect_ratio * scale
#         py = -v * scale # minus sign so that the y-axis points up

#         # ray direction 
#         D = normalize(F + px*R + py*U)
#         O = camera_position

#         # ray-sphere intersection 
#         # The equation for the sphere is given by: abs((camera_position + t * D)-sphere_position) = radius^2
#         # where t is the length of the ray. 
#         oc = O - sphere_position
#         a = dot(D,D)
#         b = 2 * dot(oc, D)
#         c = dot(oc, oc) - sphere_radius^2
#         discriminant = b^2 - 4*a*c

#         if discriminant < 0
#             image[j,i] = RGB(0,0,0)  # miss: black
#         else
#             image[j,i] = RGB(1,1,1)  # hit: white
#             sqrt_discriminant = sqrt(discriminant)
#             t1 = -b + sqrt_discriminant 
#             t2 = -b - sqrt_discriminant 
#             t_sphere = min(t1, t2) / (2 * a) # take the smallest positive t
#             closest_t[j, i] = t_sphere 
#         end

#         # ray-disc intersection 
#         if abs(D[2]) > 1e-5  
#             t_disc = - camera_position[2] / D[2] # intersection with the y=0 plane 
#             if t_disc > 0 && t_disc < closest_t[j,i]
#                 P = camera_position + t_disc * D
#                 rho = sqrt(P[1]^2 + P[3]^2) 
#                 if Rin <= rho <= Rout
#                     image[j,i] = RGB(1, 0, 0)  # hit: red
#                 end 
#             end
#         end

#         #TODO: add RK4 integrator and in each step test wheter the ray intersected the sphere or the disc 
#         #NOTE: this means that we probably can not use the DifferentialEquations.jl because we need to test 
#         # for the intersection in each step. 
#         # This could still probably be done if we set some stopping condition but i don know how it would go with parallelization.
#         #TODO: Add proper intersection functions for the sphere and the disc - since we are using spherical
#         # coordinates, this for sphere can be done with a simple distance check. 
#         #TODO: Derive the equations for the motion of one ray. Each ray represents a particle (photon) that we shoot
#         # directly from the camera. Because of the curvature it will actually be possible to see behind the black hole.
#     end 
# end 

function make_system(W::Int, H::Int)
    # This function creates array of rays with initial positions and velocities
    # that are moving away from the camera position. 
    rays = Vector{Ray}(undef, W*H)
    rays_index = 1
    for j in 1:H
        for i in 1:W 
            u = map_pixel(i, W)
            v = map_pixel(j, H)

            # apply FOV scaling 
            px = u * aspect_ratio * scale
            py = -v * scale # minus sign so that the y-axis points up

            # ray direction 
            D = normalize(F + px*R + py*U)
            # ray position 
            ray = Ray(position=camera_position, velocity=D, pixel=(i, j))
            rays[rays_index] = ray# push!(rays, ray)   # this actually leaves the first indeces undefined causing error later
            rays_index += 1
        end 
    end 
    return rays 
end
function render(rays::Array, t::Float64, dt::Float64 = 0.01)
    # This function moves the rays and renders the image
    # TODO
    # For time t ∈ [0, T] 
    # For each ray (pixel) (i, j) 
    # detect collision with the sphere and the disc, if there is a collision, the ray stops 
    # move_ray((i,j))
    println("started rendering")
    for t in 0:dt:t
        # For each ray in rays array move the ray
        # check for intersection with the sphere and the disc, if there is a collision, remove the ray from the array
        # (it needs no longer to be moved so we would be wasting time)
        for k in eachindex(rays)
            # Move each ray only if it is still active
            if rays[k].flag == 0
                move_ray(rays[k], dt)
                check_for_intersection(rays[k])
            end
        end 
    end 
    # Output image
    image = Array{RGB{Float64}}(undef, H, W)
    
    for k in eachindex(rays)
        ray = rays[k]
        if ray.flag == 0
            image[ray.pixel[1], ray.pixel[2]] = RGB(0, 0, 0)
        elseif ray.flag == 1
            # println("Ray hit")
            image[ray.pixel[1], ray.pixel[2]] = RGB(1, 1, 1)
        elseif ray.flag == 2
            image[ray.pixel[1], ray.pixel[2]] = RGB(1, 0, 0) 
        elseif ray.flag == 3 
            image[ray.pixel[1], ray.pixel[2]] = RGB(0, 1, 0)
        elseif ray.flag == 4
            image[ray.pixel[1], ray.pixel[2]] = RGB(0, 0, 1)
        end      


    end
    return image
end

function move_ray(ray::Ray, dt::Float64)
    # This function moves a single ray according to the Einstein's equations 
    # this is basically one step of the RK4 integrator
    # TODO 
    ray.position += ray.velocity * dt # this is a simple Euler step, we will replace it with RK4 later 
    
end 

function check_for_intersection(ray::Ray)
    # TODO: remove the ray form the array if it hits something for now each ray has its own flag 
    
    R = norm(ray.position - sphere_position)
    if R < sphere_radius
        # ray hit the sphere, we can stop it
        ray.flag = 1
        return 
    end
    R_second = norm(ray.position - second_sphere_position )
    if R_second < second_sphere_radius 
        ray.flag = 2 
        return 
    end 
    R_third = norm(ray.position - third_sphere_position)
    if R_third < third_sphere_radius
        ray.flag = 3
        return
    end
    if abs(ray.position[2]) < 1e-2  
        #t_disc = - camera_position[2] / ray.velocity[2] # intersection with the y=0 plane 
        #P = camera_position + t_disc * D
        rho = sqrt(ray.position[1]^2 + ray.position[3]^2) 
        if Rin <= rho <= Rout
            ray.flag = 4
            return
        end          
    end


end

function main()
    rays = make_system(W, H)
    # Move the rays
    image = render(rays, 10.0, 0.01)
    # Save the image
    save("sphere_test_ray_casting.png", image)
    println("Saved sphere_test.png")
end

main()