# Ask for a DNI number
dni = int(input("Enter your DNI number (without the letter): "))

# DNI letter table
letters = "TRWAGMYFPDXBNJZSQVHLCKE"

# Calculate the index of the letter
index = dni % 23

# Get the letter
letter = letters[index]

# Print the DNI with the letter
print(f"Your complete DNI is {dni}{letter}")