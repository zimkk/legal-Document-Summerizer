import os
import json
import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
import wandb
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Define paths for Google Colab with correct working directory
BASE_PATH = "/content/drive/MyDrive/Work/Legal"
DATASET_DIR = os.path.join(BASE_PATH, "datasets")  # Legal/datasets
MODEL_OUTPUT_DIR = os.path.join(BASE_PATH, "output")  # Legal/output
INPUT_DATASET_FILE = os.path.join(DATASET_DIR, "billsum_data.json")

# Create directories
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# Initialize wandb for experiment tracking
wandb.init(project="legal-billsum-llama3-finetuning")

# Model parameters
max_seq_length = 2048
dtype = None
load_in_4bit = True

# Check if CUDA is available
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("CUDA is not available. Please ensure you have a compatible GPU.")
    exit()

# Load the model with Unsloth optimizations
try:
    print("Loading Llama 3.1 model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/meta-llama-3.1-8B",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    print("Model loaded successfully!")
except Exception as e:
    print("Model loading failed. Please check your model name and configuration.")
    print(f"Error: {e}")
    exit()

# LoRA configuration exactly as shown in the image
try:
    print("Configuring LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    print("LoRA adapters configured successfully!")
except Exception as e:
    print("Failed to configure LoRA adapters.")
    print(f"Error: {e}")
    exit()

def load_and_prepare_dataset():
    dataset_path = INPUT_DATASET_FILE
    print(f"Loading BillSum dataset from {dataset_path}...")
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Dataset file not found at {dataset_path}. Please ensure the file exists.")
        print("Did you run the prepare_legal_dataset.py script first?")
        exit()
    
    dataset = Dataset.from_dict(data)
    
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    def formatting_prompts_func(example):
        instruction = "Summarize the following legislative bill."
        input_text = example["text"]
        response = example["summary"]
        prompt = alpaca_prompt.format(instruction, input_text, response)
        return { "text": prompt }
    
    formatted_dataset = dataset.map(formatting_prompts_func, num_proc=4)  # Reduced for Colab
    print("Dataset formatted successfully.")
    return formatted_dataset

# Training arguments optimized for Colab T4 GPU
training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    per_device_train_batch_size=1,  # Reduced for T4 GPU
    gradient_accumulation_steps=16,  # Increased to compensate for smaller batch size
    warmup_steps=5,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,  # Force fp16 for Colab
    logging_steps=5,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=3407,
    save_strategy="epoch",
    report_to="wandb",
    gradient_checkpointing=True,
)

def main():
    try:
        dataset = load_and_prepare_dataset()
        train_dataset = dataset

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            args=training_args,
            packing=True,
        )
        
        print("Starting training...")
        trainer.train()
        
        print(f"Saving final model adapters to {MODEL_OUTPUT_DIR}...")
        trainer.save_model(MODEL_OUTPUT_DIR)
        tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
        
        print("Training completed successfully!")
        print(f"Model saved to {MODEL_OUTPUT_DIR}")
        
    except Exception as e:
        print(f"An error occurred during training: {e}")
        wandb.finish()
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    main() 