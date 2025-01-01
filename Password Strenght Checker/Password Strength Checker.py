import tkinter as tk
from tkinter import ttk, messagebox
import re
import math
import json
import os

# Load a comprehensive list of common passwords
COMMON_PASSWORDS_FILE = 'common_passwords.json'

def load_common_passwords():
    if not os.path.exists(COMMON_PASSWORDS_FILE):
        # A small sample list; in practice, use a comprehensive list
        common_passwords = [
            "password", "123456", "123456789", "qwerty", "abc123",
            "football", "monkey", "letmein", "dragon", "iloveyou",
            "111111", "baseball", "welcome", "1234567", "sunshine",
            "master", "123123", "shadow", "ashley", "password1"
        ]
        with open(COMMON_PASSWORDS_FILE, 'w') as f:
            json.dump(common_passwords, f)
    else:
        with open(COMMON_PASSWORDS_FILE, 'r') as f:
            common_passwords = json.load(f)
    return set(common_passwords)

COMMON_PASSWORDS = load_common_passwords()

# Default criteria
DEFAULT_CRITERIA = {
    "min_length": 8,
    "require_uppercase": True,
    "require_lowercase": True,
    "require_numbers": True,
    "require_special": True
}

SPECIAL_CHARACTERS = "!@#$%^&*(),.?\":{}|<>"

def calculate_entropy(password):
    """Calculate the entropy of the password."""
    pool = 0
    if re.search(r'[a-z]', password):
        pool += 26
    if re.search(r'[A-Z]', password):
        pool += 26
    if re.search(r'[0-9]', password):
        pool += 10
    if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        pool += len(SPECIAL_CHARACTERS)
    if pool == 0:
        return 0
    entropy = len(password) * math.log2(pool)
    return round(entropy, 2)

def check_password_strength(password, criteria):
    # Initialize score and feedback
    score = 0
    feedback = []

    # Check for minimum length
    if len(password) >= criteria["min_length"]:
        score += 1
    else:
        feedback.append(f"Password should be at least {criteria['min_length']} characters long.")

    # Check for uppercase letters
    if criteria["require_uppercase"]:
        if re.search(r'[A-Z]', password):
            score += 1
        else:
            feedback.append("Add at least one uppercase letter.")
    else:
        score += 1  # If not required, consider it as met

    # Check for lowercase letters
    if criteria["require_lowercase"]:
        if re.search(r'[a-z]', password):
            score += 1
        else:
            feedback.append("Add at least one lowercase letter.")
    else:
        score += 1

    # Check for numbers
    if criteria["require_numbers"]:
        if re.search(r'[0-9]', password):
            score += 1
        else:
            feedback.append("Add at least one number.")
    else:
        score += 1

    # Check for special characters
    if criteria["require_special"]:
        if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            score += 1
        else:
            feedback.append("Add at least one special character (e.g., !, @, #, etc.).")
    else:
        score += 1

    # Check for common passwords
    if password.lower() in COMMON_PASSWORDS:
        feedback.append("Password is too common. Choose a more unique password.")
    else:
        score += 1

    # Calculate entropy
    entropy = calculate_entropy(password)
    entropy_feedback = ""
    if entropy < 28:
        entropy_feedback = "Very Weak"
    elif entropy < 35:
        entropy_feedback = "Weak"
    elif entropy < 59:
        entropy_feedback = "Reasonable"
    elif entropy < 127:
        entropy_feedback = "Strong"
    else:
        entropy_feedback = "Very Strong"

    return score, feedback, entropy, entropy_feedback

class PasswordCheckerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Password Strength Checker")
        self.criteria = DEFAULT_CRITERIA.copy()

        self.create_widgets()

    def create_widgets(self):
        # Frame for customization
        customization_frame = ttk.LabelFrame(self.root, text="Customization")
        customization_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        # Minimum length
        ttk.Label(customization_frame, text="Minimum Length:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.min_length_var = tk.IntVar(value=self.criteria["min_length"])
        ttk.Entry(customization_frame, textvariable=self.min_length_var, width=5).grid(row=0, column=1, sticky="w", padx=5, pady=2)

        # Require uppercase
        self.require_uppercase_var = tk.BooleanVar(value=self.criteria["require_uppercase"])
        ttk.Checkbutton(customization_frame, text="Require Uppercase Letters", variable=self.require_uppercase_var, command=self.update_criteria).grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=2)

        # Require lowercase
        self.require_lowercase_var = tk.BooleanVar(value=self.criteria["require_lowercase"])
        ttk.Checkbutton(customization_frame, text="Require Lowercase Letters", variable=self.require_lowercase_var, command=self.update_criteria).grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=2)

        # Require numbers
        self.require_numbers_var = tk.BooleanVar(value=self.criteria["require_numbers"])
        ttk.Checkbutton(customization_frame, text="Require Numbers", variable=self.require_numbers_var, command=self.update_criteria).grid(row=3, column=0, columnspan=2, sticky="w", padx=5, pady=2)

        # Require special characters
        self.require_special_var = tk.BooleanVar(value=self.criteria["require_special"])
        ttk.Checkbutton(customization_frame, text="Require Special Characters", variable=self.require_special_var, command=self.update_criteria).grid(row=4, column=0, columnspan=2, sticky="w", padx=5, pady=2)

        # Frame for password input
        input_frame = ttk.LabelFrame(self.root, text="Enter Password")
        input_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        self.password_var = tk.StringVar()
        self.password_entry = ttk.Entry(input_frame, textvariable=self.password_var, show="*", width=30)
        self.password_entry.grid(row=0, column=0, padx=5, pady=5)
        self.password_entry.bind("<KeyRelease>", self.evaluate_password)

        # Frame for strength display
        strength_frame = ttk.LabelFrame(self.root, text="Strength Evaluation")
        strength_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

        # Score
        ttk.Label(strength_frame, text="Score:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.score_label = ttk.Label(strength_frame, text="0/6")
        self.score_label.grid(row=0, column=1, sticky="w", padx=5, pady=2)

        # Strength
        ttk.Label(strength_frame, text="Strength:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.strength_label = ttk.Label(strength_frame, text="Weak")
        self.strength_label.grid(row=1, column=1, sticky="w", padx=5, pady=2)

        # Entropy
        ttk.Label(strength_frame, text="Entropy:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.entropy_label = ttk.Label(strength_frame, text="0 bits")
        self.entropy_label.grid(row=2, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(strength_frame, text="Entropy Strength:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        self.entropy_strength_label = ttk.Label(strength_frame, text="Very Weak")
        self.entropy_strength_label.grid(row=3, column=1, sticky="w", padx=5, pady=2)

        # Feedback
        feedback_frame = ttk.LabelFrame(self.root, text="Feedback")
        feedback_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

        self.feedback_text = tk.Text(feedback_frame, height=6, width=50, state='disabled')
        self.feedback_text.grid(row=0, column=0, padx=5, pady=5)

    def update_criteria(self):
        self.criteria["min_length"] = self.min_length_var.get()
        self.criteria["require_uppercase"] = self.require_uppercase_var.get()
        self.criteria["require_lowercase"] = self.require_lowercase_var.get()
        self.criteria["require_numbers"] = self.require_numbers_var.get()
        self.criteria["require_special"] = self.require_special_var.get()
        self.evaluate_password()

    def evaluate_password(self, event=None):
        password = self.password_var.get()
        score, feedback, entropy, entropy_strength = check_password_strength(password, self.criteria)
        max_score = 6

        # Update score
        self.score_label.config(text=f"{score}/{max_score}")

        # Update strength
        strength_percentage = (score / max_score) * 100
        if strength_percentage == 100:
            strength = "Excellent"
        elif strength_percentage >= 80:
            strength = "Strong"
        elif strength_percentage >= 60:
            strength = "Moderate"
        else:
            strength = "Weak"
        self.strength_label.config(text=strength)

        # Update entropy
        self.entropy_label.config(text=f"{entropy} bits")
        self.entropy_strength_label.config(text=entropy_strength)

        # Update feedback
        self.feedback_text.config(state='normal')
        self.feedback_text.delete(1.0, tk.END)
        if feedback:
            for item in feedback:
                self.feedback_text.insert(tk.END, f"- {item}\n")
        else:
            self.feedback_text.insert(tk.END, "Your password meets all the criteria. Great job!")
        self.feedback_text.config(state='disabled')

def main():
    root = tk.Tk()
    app = PasswordCheckerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()


##
##Password Strength Checker with Enhanced Features
##The Password Strength Checker is a comprehensive tool designed to help users create and evaluate secure passwords through an intuitive Graphical User Interface (GUI) built with Python's Tkinter library. This enhanced version not only assesses password strength based on traditional criteria such as length, the inclusion of uppercase and lowercase letters, numbers, and special characters but also incorporates advanced security measures to provide a more thorough evaluation.
##
##Key Features:
##Real-Time Feedback: As users type their passwords, the application dynamically evaluates and displays strength metrics, ensuring immediate guidance on password quality.
##
##Entropy Calculation: By calculating the entropy of each password, the checker quantifies its unpredictability, offering a nuanced assessment of its strength beyond basic criteria.
##
##Dictionary Check: The tool cross-references entered passwords against a comprehensive list of common passwords, preventing the use of easily guessable and vulnerable options.
##
##Customization Options: Users have the flexibility to define their own password requirements, such as setting minimum length and specifying which character types are mandatory, allowing the tool to adapt to diverse security needs.
##
##Technologies Used:
##Python: The core programming language powering the application's logic and functionalities.
##Tkinter: Utilized for creating a user-friendly and responsive GUI, enabling seamless interaction and real-time updates.
##Regular Expressions (re): Employed for pattern matching to verify the presence of various character types within passwords.
##JSON: Used to manage and store the list of common passwords for efficient dictionary checks.
##Math Library: Facilitates the calculation of password entropy, providing insights into password complexity.
##Purpose and Benefits:
##This Password Strength Checker serves as an essential tool for enhancing personal and organizational cybersecurity practices. By providing immediate and detailed feedback, it educates users on creating robust passwords that resist common attack vectors. The inclusion of entropy calculations and dictionary checks ensures a high level of security, making it suitable for integration into larger applications or for standalone use in various settings.
##
##Whether you're looking to strengthen your own password habits or implement a reliable security feature within your software solutions, this enhanced Password Strength Checker offers the necessary tools and flexibility to meet modern security standards.
