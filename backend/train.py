import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CornellMovieDataset
from model import SegawaModel, device
from tqdm import tqdm
import os
import math

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 3e-4
MAX_LENGTH = 60
VOCAB_SIZE = 8000

def train():
    print("Initializing Segawa Training Pipeline...")
    
    # 1. Load Data
    lines_path = r"datasets/movie_lines.txt"
    conv_path = r"datasets/movie_conversations.txt"
    shakespeare_path = r"datasets/input.txt"
    
    dataset = CornellMovieDataset(lines_path, conv_path, shakespeare_path, max_length=MAX_LENGTH, vocab_size=VOCAB_SIZE)
    
    # Split into Train (95%) and Validation (5%)
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    
    # 2. Initialize Model
    print(f"Loading Model to {device}...")
    model = SegawaModel(vocab_size=VOCAB_SIZE, max_length=MAX_LENGTH).to(device)
    
    # 3. Optimizer and Loss Function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.PAD_IDX)
    
    # Create checkpoints folder
    os.makedirs('checkpoints', exist_ok=True)
    
    print(f"Beginning Training Loop! (Train size: {train_size}, Val size: {val_size})")
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        
        # Training Pass
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for batch_idx, (inputs, targets) in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass (Causal mask is applied automatically in model.py)
            outputs = model(inputs)
            
            outputs = outputs.reshape(-1, outputs.shape[2])
            targets = targets.reshape(-1)
            
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}] Train")
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation Pass
        model.eval()
        total_val_loss = 0
        correct_preds = 0
        total_preds = 0
        
        with torch.no_grad():
            val_loop = tqdm(enumerate(val_loader), total=len(val_loader), leave=False)
            for batch_idx, (inputs, targets) in val_loop:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                # Loss calculation
                outputs = outputs.reshape(-1, outputs.shape[2])
                targets = targets.reshape(-1)
                
                # Calculate accuracy ignoring PAD token
                predictions = outputs.argmax(dim=1)
                mask = (targets != dataset.PAD_IDX)
                correct_preds += ((predictions == targets) & mask).sum().item()
                total_preds += mask.sum().item()
                
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
                
                val_loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}] Val")
                
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = (correct_preds / total_preds) * 100 if total_preds > 0 else 0
        
        # Calculate Perplexity (Aiming for < 50 as requested)
        val_perplexity = math.exp(avg_val_loss) if avg_val_loss < 20 else float('inf')
        
        print(f"Epoch {epoch+1} finished!")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}% | Val Perplexity: {val_perplexity:.2f}")
        
        # Stop early if perplexity target is reached
        if val_perplexity < 50.0:
            print("Target perplexity < 50 achieved!")
            
        # Save model checkpoint
        checkpoint_path = f"checkpoints/segawa_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Brain saved to {checkpoint_path}\n")

if __name__ == "__main__":
    train()
