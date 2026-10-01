from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import os
from dataset import CornellMovieDataset
from model import SegawaModel, device

app = Flask(__name__)
CORS(app) # Allow frontend to talk to backend

# Configuration
MAX_LENGTH = 60
VOCAB_SIZE = 8000
CHECKPOINT_PATH = "checkpoints/segawa_epoch_3.pth" # Change to highest epoch

print("Starting Segawa API Server...")

# 1. Load Dataset purely to get the vocabulary mappings
print("Loading vocabulary (this takes a moment)...")
lines_path = r"datasets/movie_lines.txt"
conv_path = r"datasets/movie_conversations.txt"
dataset = CornellMovieDataset(lines_path, conv_path, max_length=MAX_LENGTH, vocab_size=VOCAB_SIZE)

# 2. Initialize Model
print("Loading Segawa Brain...")
model = SegawaModel(vocab_size=VOCAB_SIZE, max_length=MAX_LENGTH).to(device)

if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print(f"Successfully loaded {CHECKPOINT_PATH}")
else:
    print(f"WARNING: Checkpoint {CHECKPOINT_PATH} not found. Running with untrained brain!")

model.eval()

def generate_response(user_input):
    words = user_input.lower().split()
    q_tokens = [dataset.word2idx.get(w, dataset.UNK_IDX) for w in words]
    
    # Start sequence with <BOS> Question <EOS>
    tokens = [dataset.BOS_IDX] + q_tokens + [dataset.EOS_IDX]
    
    input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
    
    # Autoregressive Generation
    response_words = []
    
    with torch.no_grad():
        for _ in range(MAX_LENGTH):
            # Pass current sequence through model
            output = model(input_tensor)
            
            # Get the prediction for the very last token
            next_token_logits = output[0, -1, :]
            next_token_id = next_token_logits.argmax(dim=-1).item()
            
            if next_token_id == dataset.EOS_IDX:
                break
                
            word = dataset.idx2word.get(next_token_id, '')
            if word not in ['<PAD>', '<BOS>', '<UNK>']:
                response_words.append(word)
                
            # Append next_token_id to input_tensor
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(device)
            input_tensor = torch.cat([input_tensor, next_token_tensor], dim=1)
            
            # Truncate to MAX_LENGTH to avoid embedding errors
            if input_tensor.size(1) > MAX_LENGTH:
                input_tensor = input_tensor[:, -MAX_LENGTH:]
                
    return " ".join(response_words)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
        
    try:
        response = generate_response(user_message)
        if not response.strip():
            response = "... (Segawa is struggling to find the words)"
            
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

import smtplib
from email.mime.text import MIMEText
import random
import datetime
from dotenv import load_dotenv

load_dotenv(os.path.expanduser('~/.env'))

# In-memory store for verification codes (email: {"code": "123456", "expires": datetime})
verification_codes = {}

@app.route("/send_verification", methods=["POST"])
def send_verification():
    data = request.json
    recipient_email = data.get("email")
    
    if not recipient_email:
        return jsonify({"error": "No email provided"}), 400
        
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    
    if not sender_email or not sender_password:
        return jsonify({"error": "Server email credentials not configured in ~/.env"}), 500
        
    # Generate 6 digit code
    code = str(random.randint(100000, 999999))
    verification_codes[recipient_email] = {
        "code": code,
        "expires": datetime.datetime.now() + datetime.timedelta(minutes=10)
    }
    
    # Create email
    msg = MIMEText(f"Your Segawa login code is: {code}\n\nThis code will expire in 10 minutes.")
    msg['Subject'] = 'Segawa Verification Code'
    msg['From'] = sender_email
    msg['To'] = recipient_email
    
    try:
        # Connect to Gmail SMTP (Assuming Gmail based on instructions)
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return jsonify({"message": "Verification code sent"})
    except Exception as e:
        return jsonify({"error": f"Failed to send email: {str(e)}"}), 500

@app.route("/verify_code", methods=["POST"])
def verify_code():
    data = request.json
    email = data.get("email")
    code = data.get("code")
    
    if not email or not code:
        return jsonify({"error": "Email and code required"}), 400
        
    record = verification_codes.get(email)
    
    if not record:
        return jsonify({"error": "No code requested for this email"}), 400
        
    if datetime.datetime.now() > record["expires"]:
        return jsonify({"error": "Code has expired"}), 400
        
    if record["code"] != code:
        return jsonify({"error": "Invalid code"}), 400
        
    # Verification successful
    del verification_codes[email]
    return jsonify({"message": "Verified"})

if __name__ == "__main__":
    print("Segawa Backend is running on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000)
