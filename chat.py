def chatbot(user_input):
    # This is a simple chatbot function that responds to user input.
    if "hello" in user_input.lower():
        return "Hello! How can I assist you today?"
    elif "how are you" in user_input.lower():
        return "I'm just a chatbot, but I'm doing great! Thanks for asking."
    elif "bye" in user_input.lower():
        return "Goodbye! Have a great day!"
    else:
        return "I'm sorry, I didn't understand that. Can you please rephrase?"
    
while True:
    msg=input("You:")
    print("Chatbot:",chatbot(msg))