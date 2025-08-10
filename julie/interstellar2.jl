# TODO: 
# The black hole will be centered at the origin.
# We will use the Schwarzschild metric to describe 
using LinearAlgebra
using Colors
using FileIO
using ImageIO
# Parameters 
# image size 
const W = 200 # horizontal resolution 
const H = 200 # vertical resolution 
# camera
const camera_position = [0.0, 1.0, -3.0] # camera position in space
const fov = π/3 # field of view
const look_at = [0.0, 0.0, 0.0] # point the camera is looking at 
const world_up = [0.0, 1.0, 0.0] # world up vector 
# sphere and disc
const sphere_position = [0.0, 0.0, 0.0] # center of the sphere 
const sphere_radius = 0.5 # radius of the sphere 
const Rin = 0.6 # inner radius of the disc 
const Rout = 0.8 # outer radius of the disc 
# camera basics 
const F = normalize(look_at - camera_position) # where the camera is looking at 
const R = normalize(cross(F, world_up)) # horizontal axis of the camera
const U = normalize(cross(R, F)) # vertical axis of the camera -- note that U should be already normalized, we do it for convenience
const aspect_ratio = W / H # aspect ration 
const scale = tan(fov / 2) 

@inline function map_pixel(x::Int, N::Int)::Float64
    return ((x - 0.5) / N - 0.5) * 2.0
end

# Output image
image = Array{RGB{Float64}}(undef, H, W)
# Stores the closest intersection distance for each pixel 
closest_t = fill(Inf, W, H)

for j in 1:H
    for i in 1:W 
        # normalize pixel coordinates
        u = map_pixel(i, W)
        v = map_pixel(j, H)

        # apply FOV scaling 
        px = u * aspect_ratio * scale
        py = -v * scale # minus sign so that the y-axis points up

        # ray direction 
        D = normalize(F + px*R + py*U)
        O = camera_position

        # ray-sphere intersection 
        # The equation for the sphere is given by: abs((camera_position + t * D)-sphere_position) = radius^2
        # where t is the length of the ray. 
        oc = O - sphere_position
        a = dot(D,D)
        b = 2 * dot(oc, D)
        c = dot(oc, oc) - sphere_radius^2
        discriminant = b^2 - 4*a*c

        if discriminant < 0
            image[j,i] = RGB(0,0,0)  # miss: black
        else
            image[j,i] = RGB(1,1,1)  # hit: white
            sqrt_discriminant = sqrt(discriminant)
            t1 = -b + sqrt_discriminant 
            t2 = -b - sqrt_discriminant 
            t_sphere = min(t1, t2) / (2 * a) # take the smallest positive t
            closest_t[j, i] = t_sphere 
        end

        # ray-disc intersection 
        if abs(D[2]) > 1e-5  
            t_disc = - camera_position[2] / D[2] # intersection with the y=0 plane 
            if t_disc > 0 && t_disc < closest_t[j,i]
                P = camera_position + t_disc * D
                rho = sqrt(P[1]^2 + P[3]^2) 
                if Rin <= rho <= Rout
                    image[j,i] = RGB(1, 0, 0)  # hit: red
                end 
            end
        end
    end 
end 

save("sphere_test1.png", image)
println("Saved sphere_test.png")