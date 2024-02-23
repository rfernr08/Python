# Ask for a number
num = int(input("Enter a number: "))

# Initialize factorial to 1
factorial = 1

if num > 0:
    # Calculate factorial
    for i in range(1, num + 1):
        factorial *= i

# Print the factorial
print(f"The factorial of {num} is {factorial}")