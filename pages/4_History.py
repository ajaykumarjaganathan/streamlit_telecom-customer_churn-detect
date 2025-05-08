import streamlit as st
import json

class History:
    def __init__(self):
        self.history = []

    def add_entry(self, entry):
        self.history.append(entry)

    def clear_history(self):
        self.history = []

    def display_history(self):
        st.title("History")
        if not self.history:
            st.write("No history yet.")
        else:
            history_data = [{"Entry": f"{i+1}. {entry}"} for i, entry in enumerate(reversed(self.history))]
            st.table(history_data)

    def save_history_to_file(self, filename):
        with open(filename, "w") as file:
            json.dump(self.history, file)

    def load_history_from_file(self, filename):
        try:
            with open(filename, "r") as file:
                self.history = json.load(file)
        except FileNotFoundError:
            st.warning(f"History file '{filename}' not found.")

# Create a shared instance of the History class
history = History()

# Log the page visits and actions
def log_history(page, action):
    history.add_entry(f"Visited page '{page}' and performed action: '{action}'")

# Example usage:
log_history("Home", "Clicked on 'Login' button")
log_history("Data", "Filtered the dataset")

# Display the history in the Streamlit app
history.display_history()
