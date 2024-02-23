numbers = []
squares = []

while True:
    try:
        # Ask for a number
        num = float(input("Enter a number: ")) 
        # If the number is negative, break the loop
        if num < 0:
            numbers.append(num)
            squares.append(num**2)
            break
        # Otherwise, add the number and its square to the respective lists
        numbers.append(num)
        squares.append(num**2)
    except ValueError:
        print("That's not a valid number. Please try again.")

# Print the numbers, their squares, and the sum of the squares
print("Numbers: ", [num for num in numbers])
print("Squares: ", [sq for sq in squares])
print("Sum of squares: ", sum(squares))