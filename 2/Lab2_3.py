import math

def add_values(*values):
    return sum(values)

def subtract_values(a, b):
    return a - b

def multiply_values(*values):
    result = 1
    for value in values:
        result *= value
    return result

def divide_values(a, b):
    return a / b

def power_value(a, b):
    return a ** b

def natural_logarithm(a):
    return math.log(a)

def main():
    while True:
        print("1. Add values")
        print("2. Subtract two values")
        print("3. Multiply values")
        print("4. Divide two values")
        print("5. Calculate power")
        print("6. Calculate natural logarithm")
        print("7. Exit")
        choice = int(input("Choose an option: "))

        if choice == 1:
            values = list(map(float, input("Enter values separated by space: ").split()))
            print("Result: ", add_values(*values))
        elif choice == 2:
            a, b = map(float, input("Enter two values separated by space: ").split())
            print("Result: ", subtract_values(a, b))
        elif choice == 3:
            values = list(map(float, input("Enter values separated by space: ").split()))
            print("Result: ", multiply_values(*values))
        elif choice == 4:
            a, b = map(float, input("Enter two values separated by space: ").split())
            print("Result: ", divide_values(a, b))
        elif choice == 5:
            a, b = map(float, input("Enter two values separated by space: ").split())
            print("Result: ", power_value(a, b))
        elif choice == 6:
            a = float(input("Enter a value: "))
            print("Result: ", natural_logarithm(a))
        elif choice == 7:
            break
        else:
            print("Invalid option")

if __name__ == "__main__":
    main()