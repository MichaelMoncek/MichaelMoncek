function vectoradd(a, b, indices)
    c = similar(a)
    for i in indices 
        c[i] = a[i] + b[i]
    end
    return c
end

function count_greater_than_50_branching(arr)
    count = 0
    for x in arr 
        if x > 50 
            count += 1
        end
    end 
    return count
end

function count_greater_than_50_branchless(arr) 
    count = 0
    for x in arr 
        count += (x > 50)
    end 
    return count
end

function normalise_div!(out, arr)
    total = sum(arr) 
    out.= arr ./ total
end

function normalise_mul!(out, arr)
    inv_total = inv(sum(arr))
    out .= arr.* inv_total
end

function mydot(a,b)
    s = zero(eltype(a))
    for i in eachindex(a, b)
        s += a[i] * b[i]
    end 
    return s
end

function mydot_simd(a,b)
    s = zero(eltype(a))
    @simd for i in eachindex(a, b)
        s += a[i] * b[i]
    end 
    return s
end

function mysum(a)
    s = zero(eltype(a))
    for i in eachindex(a)
        s += a[i]
    end 
    return s
end

function mysum_simd(a)
    s = zero(eltype(a))
    @simd for i in eachindex(a)
        s += a[i]
    end 
    return s
end

function test_slice(x)
    z = x[:, 1]
    return sum(z)
end

function test_view(x)
    z = @views x[:, 1]
    return sum(z)
end