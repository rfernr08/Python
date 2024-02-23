# Ask for a sentence
sentence = input("Enter a sentence: ")

# Give the user choices
print("Choose an option:")
print("1. Convert the sentence into uppercase")
print("2. Convert the sentence into lowercase")
print("3. Convert the first character of each word into uppercase")
print("4. Convert the characters that are in even positions into uppercase")

# Get the user's choice
choice = input("Your choice: ")

if choice == '1':
    result = sentence.upper()
elif choice == '2':
    result = sentence.lower()
elif choice == '3':
    result = sentence.title()
elif choice == '4':
    result = ""
    for i in range(0,len(sentence)):
        if i % 2 == 0:
            result += sentence[i].upper()
        else:
            result += sentence[i]

else:
    result = "Invalid choice"

# Print the result
print(result)