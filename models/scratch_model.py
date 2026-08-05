import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# Hyperparameters for the scratch model (small for CPU training demo)
BLOCK_SIZE = 128    # Maximum context length for predictions
N_EMBD = 128        # Embedding dimension
N_HEAD = 4          # Number of attention heads
N_LAYER = 2         # Number of Transformer layers
DROPOUT = 0.1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class CharTokenizer:
    """Character-level tokenizer for encoding and decoding code strings."""
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

class Head(nn.Module):
    """One head of self-attention."""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * (C**-0.5) # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # Perform the weighted aggregation of the values
        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """A simple linear layer followed by a non-linearity."""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: communication followed by computation."""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class ScratchTransformer(nn.Module):
    """Decoder-only Transformer model for character-level code generation."""
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(N_EMBD, n_head=N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD) # Final layer norm
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensors of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.7, top_k=None):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last BLOCK_SIZE tokens
            idx_cond = idx[:, -BLOCK_SIZE:]
            # Get predictions
            logits, loss = self(idx_cond)
            # Focus only on the last time step logits
            logits = logits[:, -1, :] / temperature # (B, vocab_size)
            
            # Optional: Top-K filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# Helper function to load dataset and initialize tokenizer
def load_data(filepath):
    if not os.path.exists(filepath):
        # Fallback inline dataset if file doesn't exist
        text = "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n"
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
    return text

def get_batch(data_tensor, batch_size=32):
    ix = torch.randint(len(data_tensor) - BLOCK_SIZE, (batch_size,))
    x = torch.stack([data_tensor[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data_tensor[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

def train_scratch_model(dataset_path, epochs=1000, lr=1e-3, batch_size=32, progress_callback=None):
    """Trains the scratch Transformer model and streams progress via progress_callback.
    
    progress_callback: function that accepts (epoch, loss, generated_sample)
    """
    text = load_data(dataset_path)
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    
    # Split data
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    model = ScratchTransformer(tokenizer.vocab_size).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(1, epochs + 1):
        model.train()
        xb, yb = get_batch(train_data, batch_size)
        
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Every 50 epochs or on the first/last epoch, evaluate and sample
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                # Estimate val loss
                val_xb, val_yb = get_batch(val_data, batch_size)
                _, val_loss = model(val_xb, val_yb)
                
                # Generate sample snippet starting with a newline or 'def '
                context = torch.tensor([tokenizer.encode("def ")], dtype=torch.long, device=DEVICE)
                generated_tokens = model.generate(context, max_new_tokens=100)[0].tolist()
                sample_text = tokenizer.decode(generated_tokens)
                
                if progress_callback:
                    progress_callback(epoch, float(loss.item()), float(val_loss.item()), sample_text)
                    
    # Save the model state dict and tokenizer metadata
    save_path = os.path.join(os.path.dirname(dataset_path), 'scratch_model.pt')
    meta_path = os.path.join(os.path.dirname(dataset_path), 'scratch_meta.pt')
    torch.save(model.state_dict(), save_path)
    torch.save({
        'chars': tokenizer.chars,
        'stoi': tokenizer.stoi,
        'itos': tokenizer.itos
    }, meta_path)
    
    return model, tokenizer

def load_saved_scratch_model(model_dir_path):
    save_path = os.path.join(model_dir_path, 'scratch_model.pt')
    meta_path = os.path.join(model_dir_path, 'scratch_meta.pt')
    if not (os.path.exists(save_path) and os.path.exists(meta_path)):
        return None, None
        
    meta = torch.load(meta_path)
    tokenizer = CharTokenizer("")
    tokenizer.chars = meta['chars']
    tokenizer.vocab_size = len(tokenizer.chars)
    tokenizer.stoi = meta['stoi']
    tokenizer.itos = meta['itos']
    
    model = ScratchTransformer(tokenizer.vocab_size).to(DEVICE)
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    model.eval()
    return model, tokenizer

if __name__ == '__main__':
    # Test execution
    print("Testing scratch model...")
    dataset_file = "../data/python_dataset.txt"
    if not os.path.exists(dataset_file):
        dataset_file = "data/python_dataset.txt"
        
    def test_cb(epoch, train_loss, val_loss, sample):
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"--- Sample Output ---\n{sample}\n---------------------")
        
    train_scratch_model(dataset_file, epochs=100, lr=1e-3, batch_size=16, progress_callback=test_cb)
