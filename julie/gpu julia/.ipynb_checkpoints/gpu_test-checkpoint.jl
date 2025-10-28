using AMDGPU

# function vadd!(c, a, b)
#     i = workitemIdx().x + (workgroupIdx().x - 1) * workgroupDim().x
#     if i <= length(a)
#         c[i] = a[i] + b[i]
#     end
#     return
# end

# a = AMDGPU.ones(Int, 1024)
# b = AMDGPU.ones(Int, 1024)
# c = AMDGPU.zeros(Int, 1024)

# groupsize = 256
# gridsize = cld(length(c), groupsize)
# @roc groupsize=groupsize gridsize=gridsize vadd!(c, a, b)
# @assert (a .+ b) ≈ c

AMDGPU.versioninfo()
@assert AMDGPU.functional()

# CPU based operation
N = 2048
a = rand(Float32, N);
b = rand(Float32, N);
c = similar(a);

c .= a .+ b;
# a_gpu = AMDGPU.roc(a) 
a_gpu = roc(a)
b_gpu = roc(b)
c_gpu = similar(a_gpu)

c_gpu .= a_gpu .+ b_gpu

# Try to see if equal
isapprox(c, c_gpu)

# Copy back from the GPU 
c_from_gpu = Array(c_gpu)

# Check whether they are equal 
isapprox(c, c_from_gpu)