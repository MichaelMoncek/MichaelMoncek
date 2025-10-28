module NeuralNetworkExample
# Simple two-layer perceptron for function approximation
mutable struct TwoLayerNet
    W1::Matrix{Float64}  # First layer weights
    b1::Vector{Float64}  # First layer biases
    W2::Matrix{Float64}  # Second layer weights
    b2::Vector{Float64}  # Second layer biases
    hidden_size::Int
end

function initialise_network(input_size::Int, hidden_size::Int, output_size::Int)
    # Xavier initialization
    W1 = randn(hidden_size, input_size) * sqrt(2.0 / input_size)
    b1 = zeros(hidden_size)
    W2 = randn(output_size, hidden_size) * sqrt(2.0 / hidden_size)
    b2 = zeros(output_size)
    
    return TwoLayerNet(W1, b1, W2, b2, hidden_size)
end

relu(x) = max(0.0, x)
relu_derivative(x) = x > 0.0 ? 1.0 : 0.0
function compute_loss(predictions, targets)
    0.5 * sum(x->x^2, predictions .- targets) / size(predictions, 2)
end

function forward_pass(net::TwoLayerNet, X::Matrix{Float64})    
    # First layer
    Z1 = net.W1 * X .+ net.b1
    A1 = relu.(Z1)
    
    # Second layer (output)
    Z2 = net.W2 * A1 .+ net.b2
    
    return Z1, A1, Z2
end

function backward_pass(net::TwoLayerNet, X::Matrix{Float64}, Z1::Matrix{Float64}, 
                      A1::Matrix{Float64}, Z2::Matrix{Float64}, targets::Matrix{Float64})
    # Backpropagation - most computationally expensive stage
    batch_size = size(X, 2)
    
    # Output layer gradients
    dZ2 = (Z2 .- targets) ./ batch_size
    dW2 = dZ2 * A1'
    db2 = sum(dZ2, dims=2)
    
    # Hidden layer gradients
    dA1 = net.W2' * dZ2
    dZ1 = dA1 .* relu_derivative.(Z1)
    dW1 = dZ1 * X'
    db1 = sum(dZ1, dims=2)
    
    return dW1, db1, dW2, db2
end

function update_parameters!(net::TwoLayerNet, dW1, db1, dW2, db2, learning_rate::Float64)
    # Parameter updates - low computational cost
    net.W1 .-= learning_rate .* dW1
    net.b1 .-= learning_rate .* db1
    net.W2 .-= learning_rate .* dW2
    net.b2 .-= learning_rate .* db2
end

function generate_training_data(n_samples::Int)
    # Generate synthetic non-linear data
    X = randn(1, n_samples)
    Y = sin.(X) .+ 0.5 .* cos.(2 .* X) .+ 0.1 .* randn(1, n_samples)
    return X, Y
end

function train_network(n_samples::Int, n_epochs::Int, hidden_size::Int=50; should_print=false)
    # Main training loop
    X, Y = generate_training_data(n_samples)
    net = initialise_network(1, hidden_size, 1)
    learning_rate = 0.01
    
    losses = Float64[]
    
    for epoch in 1:n_epochs
        # Forward pass
        Z1, A1, Z2 = forward_pass(net, X)
        
        # Compute loss
        loss = compute_loss(Z2, Y)
        push!(losses, loss)
        
        # Backward pass
        dW1, db1, dW2, db2 = backward_pass(net, X, Z1, A1, Z2, Y)
        
        # Update parameters
        update_parameters!(net, dW1, db1, dW2, db2, learning_rate)
        
        # Print progress occasionally
        if should_print && epoch % 100 == 0
            println("Epoch $epoch, Loss: $loss")
        end
    end
    
    return net, losses
end

end #module


using Profile

# First run the function to compile it (on small sizes)
NeuralNetworkExample.train_network(10, 2, 10)

# Clear any previous profiling data
Profile.clear()

# Run the profiler on a longer training session, inside the VS Code REPL
@profview NeuralNetworkExample.train_network(1000, 10000, 20)