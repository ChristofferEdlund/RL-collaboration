



x = glucose
b = bolus_edition
x = (1, 1, 1, 1, 1.3)
b = (0, 0, 0.5, 0, 0, 0.8)

input = x+b
output = x

prediction_paris = [input[i], output[i+1]]

inte = integrations(x, b)
data = generate_pairs(x, b, inte)
data.shuffle()

current_value = input[0]

for d in range(data):
    x, y = data

    ode_int = model(x)
    loss(ode_int, y)



def generate_pairs(x, b, inte):
    # x <- (1, 10)
    # b <- (1, 10)
    # inte <- (1, 9)

    data = []
    for i in range(len(x)-1):

        data.append(((x+b)[i], inte[i]))

    # data <- (9, 2)
    return data