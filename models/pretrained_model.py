import sys
import os
import torch

class PretrainedCodeGenerator:
    """Manages the download, caching, and inference of a pre-trained code LLM using HuggingFace."""
    def __init__(self, model_name="Qwen/Qwen2.5-Coder-0.5B-Instruct"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self):
        """Lazy load the model to avoid overhead at startup."""
        if self.model is not None:
            return
            
        print(f"Loading pre-trained model '{self.model_name}' on {self.device}...", flush=True)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("Hugging Face 'transformers' is not installed. Please run pip install transformers.")

        # Download and cache model locally
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32, # CPU friendly
            device_map="auto"
        )
        print("Model loaded successfully!", flush=True)

    def generate(self, prompt, max_new_tokens=256, temperature=0.7, top_k=50):
        """Generates code from a prompt using Qwen Chat Template conventions."""
        self.load_model()
        
        # Format input using Qwen chat structures for best results
        system_msg = "You are an expert AI software engineer. Generate clean, efficient, and well-documented Python code. Return ONLY Python code. Do not write introductory or concluding remarks."
        
        # If Qwen-Coder model, apply the template
        if "Qwen" in self.model_name:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback for GPT2 or other base models
            text = f"# Python code for: {prompt}\n"

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        # Slice inputs to get only the newly generated tokens
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_len:]
        
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip()

if __name__ == '__main__':
    # Test generation with a simple prompt
    print("Testing pretrained model (Note: This will download Qwen2.5-Coder-0.5B-Instruct if not cached)...")
    generator = PretrainedCodeGenerator()
    try:
        # We run a small generation to verify
        code = generator.generate("Write a function to add two numbers.", max_new_tokens=50)
        print("\n=== Generated Code ===\n")
        print(code)
        print("\n======================\n")
    except Exception as e:
        print(f"Error during pretrained generation test: {e}")
        print("This is normal if offline or during light verification tests.")
