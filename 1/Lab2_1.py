# Ask for temperature in Celsius
celsius = float(input("Enter temperature in Celsius: "))

# Convert to Fahrenheit
fahrenheit = (celsius * 9/5) + 32

celsius = round(celsius, 1)
fahrenheit = round(fahrenheit, 1)
# Print both temperatures with 1 decimal place
print(f"The temperature of {celsius} degrees Celsius corresponds to {fahrenheit} degrees Fahrenheit")