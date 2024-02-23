import math
import random
import json
import os

def load_scores():
    if os.path.exists('scores.json'):
        with open('scores.json', 'r') as f:
            return json.load(f)
    else:
        return {}

def save_scores(scores):
    with open('scores.json', 'w') as f:
        json.dump(scores, f)

def play_game():
    scores = load_scores()

    name = input("Enter your name: ")
    N = int(input("Enter a number N: "))

    print(f"Welcome {name}, your highest score is: {scores.get(name, 0)}")

    number_to_guess = random.randint(1, N)
    max_trials = math.ceil(N)
    trials = 0

    while trials < max_trials:
        guess = int(input("Guess the number: "))
        trials += 1

        if guess < number_to_guess:
            print("Higher")
        elif guess > number_to_guess:
            print("Lower")
        else:
            points = N // (2 * trials)
            print(f"Congratulations {name}! You've guessed the number. Your score is: {points}")
            if points > scores.get(name, 0):
                scores[name] = points
                save_scores(scores)
            break

        print(f"Trials left: {max_trials - trials}")

    else:
        print("Sorry, you didn't guess the number.")

def show_scores():
    scores = load_scores()
    for name, score in scores.items():
        print(f"{name}: {score}")

def main():
    while True:
        print("1. Play game")
        print("2. Show scores")
        print("3. Exit")
        choice = int(input("Choose an option: "))

        if choice == 1:
            play_game()
        elif choice == 2:
            show_scores()
        elif choice == 3:
            break
        else:
            print("Invalid option")

if __name__ == "__main__":
    main()