import os
import google.generativeai as genai

def generate_chat_response(api_key, prompt, history, model_name="gemini-1.5-flash"):
    """
    Sends a message to the Gemini API, maintaining the conversation history.
    
    history: List of dicts, e.g., [{"role": "user"|"model", "text": "..."}]
    """
    if not api_key:
        raise ValueError("Gemini API Key is missing. Please configure it in the dashboard settings.")
        
    # Configure the client
    genai.configure(api_key=api_key)
    
    # Map frontend roles to official Gemini roles ('user' or 'model')
    gemini_history = []
    for turn in history:
        role = turn.get("role", "user")
        # Handle different potential role formats from frontends
        if role in ["assistant", "model"]:
            gemini_role = "model"
        else:
            gemini_role = "user"
            
        gemini_history.append({
            "role": gemini_role,
            "parts": [turn.get("text", "")]
        })
        
    # Initialize the generative model
    # gemini-1.5-flash and gemini-2.5-flash both support system instructions
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=(
            "You are CodeArchitect Chatbot, a premium coding companion. "
            "Help the user write clean, correct, and well-commented code. "
            "Answer programming questions concisely and explain concepts clearly. "
            "Use Markdown formatting with syntax-highlighted code blocks for any code outputs."
        )
    )
    
    # Start chat with the prepared history
    chat = model.start_chat(history=gemini_history)
    
    # Send the new user prompt
    response = chat.send_message(prompt)
    return response.text
