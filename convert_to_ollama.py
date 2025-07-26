import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def convert_to_ollama_format():
    # Paths for local use
    current_dir = os.getcwd()
    input_model_path = os.path.join(current_dir, "output")  # Your downloaded fine-tuned model
    output_model_path = os.path.join(current_dir, "ollama_model")
    
    print("Loading base model and adapters...")
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "unsloth/meta-llama-3.1-8B",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load the LoRA adapters
    model = PeftModel.from_pretrained(
        base_model,
        input_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("Merging adapters with base model...")
    # Merge LoRA weights with base model
    merged_model = model.merge_and_unload()
    
    print("Saving merged model in Ollama format...")
    # Create output directory
    os.makedirs(output_model_path, exist_ok=True)
    
    # Save the model in a format Ollama can use
    merged_model.save_pretrained(
        output_model_path,
        safe_serialization=True
    )
    
    # Save the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(input_model_path)
    tokenizer.save_pretrained(output_model_path)
    
    print(f"Model converted and saved to {output_model_path}")
    print("\nTo use with Ollama:")
    print("1. The Modelfile is already configured to use:", output_model_path)
    print("2. Run: ollama create legal-summarizer -f Modelfile")
    print("3. Test with: ollama run legal-summarizer 'Your legal text here...'")

if __name__ == "__main__":
    convert_to_ollama_format() 