from pylie import SE3 

# Random pose
T = SE3.random()

# R^n to group directly (using "Capital" notation)
T = SE3.Exp([0.1, 0.2, 0.3, 4, 5, 6])

# Group to R^n directly
x = SE3.Log(T)

# Wedge, vee
Xi = SE3.wedge(x)
x = SE3.vee(x)

# Actual exp/log maps 
T = SE3.exp(Xi)
Xi = SE3.log(T)



