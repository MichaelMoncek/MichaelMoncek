using Images, Plots, FFTW

# Grid size and parameters 
const GRID_SIZE = 128#128  # Grid resolution
const TIME_STEPS = 100  # Number of iterations
const dt = 0.15  # Time step size

# Function to create a Gaussian kernel
function gaussian_kernel(size::Int, sigma::Float64)
    center = size ÷ 2
    kernel = [exp(-((x - center)^2 + (y - center)^2) / (2 * sigma^2)) for x in 1:size, y in 1:size]
    return kernel ./ sum(kernel)  # Normalize kernel
end

# Function to create a Gaussian assymetric kernel 
function asymmetric_kernel(size, sigma, shift_x, shift_y)
    kernel = gaussian_kernel(size, sigma)
    return circshift(kernel, (shift_x, shift_y))  # Shift kernel in one direction
end

# Function to apply convolution using FFT
function apply_kernel(grid, kernel)
    grid_fft = fft(grid)
    # Manually pad the kernel to the same size as the grid
    padded_kernel = zeros(eltype(kernel), size(grid))
    padded_kernel[1:size(kernel, 1), 1:size(kernel, 2)] = kernel
    kernel_fft = fft(padded_kernel)
    convolved = real(ifft(grid_fft .* kernel_fft))
    return circshift(convolved, (-size(kernel, 1) ÷ 2, -size(kernel, 2) ÷ 2))
end

# Growth function for Lenia
function growth_function(x, mu=0.5, sigma=0.1076)#sigma=0.017)
    return exp(-((x - mu)^2) / (2 * sigma^2)) - 0.5
end

function init_orbium!(grid; scale=1.0)
    # Base Orbium pattern (5x5) - will be scaled up
    base_pattern = [
        0.0  0.2  0.5  0.2  0.0;
        0.2  0.8  1.0  0.8  0.2;
        0.5  1.0  1.0  1.0  0.5;
        0.2  0.8  1.0  0.8  0.2;
        0.0  0.2  0.5  0.2  0.0;
    ]
    base_pattern = base_pattern
    # Scale up the pattern
    scaled_size = Int(round(size(base_pattern, 1) * scale))  # Scaled-up size
    pattern_resized = zeros(scaled_size, scaled_size)

    for i in 1:scaled_size, j in 1:scaled_size
        # Map larger pattern coordinates to smaller base pattern
        base_i = round(Int, (i - 1) / (scaled_size - 1) * (size(base_pattern, 1) - 1)) + 1
        base_j = round(Int, (j - 1) / (scaled_size - 1) * (size(base_pattern, 2) - 1)) + 1
        pattern_resized[i, j] = base_pattern[base_i, base_j]
    end

    cx, cy = size(grid,1) ÷ 2, size(grid,2) ÷ 2
    px, py = size(pattern_resized,1) ÷ 2, size(pattern_resized,2) ÷ 2

    # Define relative positions for five seeds
    offsets = [
        (0, 0),        # Center
        (5,5)#(-30, -30),  # Top-left
        #(30, 30),    # Bottom-right
        #(-30, 30),   # Bottom-left
        #(30, -30),    # Top-right
    ]

    # Place all five seeds safely within the grid
    for (ox, oy) in offsets
        x_start = clamp(cx - px + 1 + ox, 1, size(grid,1) - size(pattern_resized,1) + 1)
        y_start = clamp(cy - py + 1 + oy, 1, size(grid,2) - size(pattern_resized,2) + 1)
        x_end = x_start + size(pattern_resized,1) - 1
        y_end = y_start + size(pattern_resized,2) - 1

        grid[x_start:x_end, y_start:y_end] .+= pattern_resized # .+ rand()
    end

        
        # # Find the center of the grid and insert the pattern
        # cx, cy = size(grid,1) ÷ 2, size(grid,2) ÷ 2
        # px, py = size(pattern_resized,1) ÷ 2, size(pattern_resized,2) ÷ 2
        # # First seed (center)
        # grid[cx-px+1:cx-px+size(pattern_resized,1), cy-py+1:cy-py+size(pattern_resized,2)] .= pattern_resized
        # # Second seed (offset position)
        # offset_x, offset_y = 50, 50  # Adjust offset for second seed placement
        # grid[cx-px+1+offset_x:cx-px+size(pattern_resized,1)+offset_x, cy-py+1+offset_y:cy-py+size(pattern_resized,2)+offset_y] .= pattern_resized
end

# Orbium kernel
function orbium_kernel(size::Int, radius::Float64)
    kernel = zeros(size, size)
    center = size ÷ 2 + 1
    
    for i in 1:size, j in 1:size
        dist = sqrt((i - center)^2 + (j - center)^2)
        kernel[i, j] = exp(-((dist - radius)^2) / (2 * (radius / 3)^2))  # Gaussian peak
    end
    
    return kernel ./ sum(kernel)  # Normalize the kernel
end



# Initialize grid with a small circular seed
function initialize_grid(grid_size)
    grid = zeros(grid_size, grid_size)
    center = grid_size ÷ 2
    # for x in center-5:center+5, y in center-5:center+5
    #     if (x - center)^2 + (y - center)^2 < 25  # Circular seed
    #         grid[x, y] = 1.0
    #     end
    # end

    init_orbium!(grid, scale=6.0) #3.0
    # orbium_kernel(grid_size, 35.0)
    return grid
end

# Run the simulation
function run_lenia()
    grid = initialize_grid(GRID_SIZE)
    # kernel = gaussian_kernel(151, 50.0)  # Kernel size and spread
    kernel = asymmetric_kernel(10, 2.0, 2, 3)  # Shifted left
    anim = @animate for t in 1:TIME_STEPS
        #kernel = asymmetric_kernel(22, 10.9, -6-round(3*cos(t/TIME_STEPS*pi*2)), -6-round(3*sin(t/TIME_STEPS*pi*2)))
        field = apply_kernel(grid, kernel)
        growth = growth_function.(field)
        grid += dt * (2 * growth .- 1)  # Update rule
        grid = clamp.(grid, 0, 1)  # Keep values in [0,1]
        heatmap(grid, title="Lenia Simulation (t=$t)", color=:viridis)
    end
    gif(anim, "lenia_simulation.gif", fps=60) #20
end

# Run the simulation
run_lenia()
