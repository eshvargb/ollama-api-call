import requests

# Set up your model and API URL
MODEL_NAME = 'gemma3:4b'
OLLAMA_API_URL = 'http://localhost:11434/api/generate'

def ask_ollama(prompt, model=MODEL_NAME):
    """
    Send a prompt to Ollama and return the response.
    """
    payload = {
        'model': model,
        'prompt': prompt,
        'stream': False  # Set to True for streamed responses
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('response', '')
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print("Request failed:", e)
        return None


def generate_related_questions(user_question):
    """
    Constructs a prompt to get related questions from the model.
    """
    prompt = f"""
You are an expert at creating simple, related questions based on a given input. 
Your task is to generate 5 follow-up questions that are closely related to the user's original question. 
These should be relevant, easy-to-understand questions that a user might naturally ask next.

Instructions:
- Only provide the questions, no explanations or additional text.
- Format them as a numbered list (1. 2. 3. 4. 5.).

Original question:
"{user_question}"

Now generate 5 related questions:
"""
    return ask_ollama(prompt)



def main():
    print("=== Related Question Generator ===")
    
    user_question = input("Enter your question: ").strip()
    
    if not user_question:
        print("You need to enter a question!")
        return
    
    print("\nGenerating related questions...\n")
    
    related = generate_related_questions(user_question)
    
    if related:
        print("Here are 5 related questions:\n")
        print(related)
    else:
        print("Failed to get a response from the model.")


if __name__ == '__main__':
    main()
