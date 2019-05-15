using Random, Distributions, Printf

Random.seed!(10)

# Input data.
X = [0 0; 0 1; 1 0; 1 1]
X_ = hcat(X, -ones(4, 1))  # Bias-attached input.
y = [0 1 1 0]

# Define learning rate.
LEARNING_RATE = 0.25

# Initialize weights.
w = Random.rand(Distributions.Normal(), (3, 1))

function forward(input, weights)
    activations = input * weights
    # macro '@.' is used when we want element-wise operation
    # for the rest of the expression.
    return @. ifelse(activations > 0, 1, 0)
end

function update_weight(input, weights, targets)
    activations = forward(input, weights)
    weights -= LEARNING_RATE * (input' * (activations - targets'))
    return weights
end

for i in 1:50
    global X_, w, y
    w = update_weight(X_, w, y)
    @printf("[Iteration %2d] Prediction %s, weight %s\n", i, forward(X_, w), w)
end