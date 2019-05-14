using Random, Distributions, Printf

Random.seed!(10)

# Input data.
X = [0 0; 0 1; 1 0; 1 1]
y = [0 1 1 1]

# Define learning rate.
LEARNING_RATE = 0.25

# Initialize weights.
w = Random.rand(Distributions.Normal(), (3, 1))

function forward(input, weights)
    activations = hcat(input, -ones(4, 1)) * weights
    # macro '@.' is used when we want element-wise operation
    # for the rest of the expression.
    return @. ifelse(activations > 0, 1, 0)
end

function update_weight(input, weights, targets)
    activations = forward(input, weights)
    weights -= LEARNING_RATE * (hcat(input, -ones(4, 1))' * (activations - targets'))
    return weights
end

for i in 1:10
    global X, w, y
    w = update_weight(X, w, y)
    @printf("==Iteration %d==\n", i)
    println(w)
    println(forward(X, w))
    println()
end