using AMDGPU
using LinearAlgebra
using BenchmarkTools
LinearAlgebra.BLAS.set_num_threads(1)

N = 2048
A = rand(Float32, N, N);
B = rand(Float32, N, N);
C = similar(A);

@benchmark mul!($C, $A, $B)

A_gpu = roc(A);
B_gpu = roc(B);
C_gpu = similar(A_gpu);

@benchmark AMDGPU.@sync mul!($C_gpu, $A_gpu, $B_gpu)