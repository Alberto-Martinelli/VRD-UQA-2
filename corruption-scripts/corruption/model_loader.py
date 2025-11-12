import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import ollama


class ModelLoader:
    _instance = None
    model = None
    tokenizer = None
    model_provider = None
    model_name = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self, model_provider, model_name):
        # Only load if model is None or if model/provider changed
        if (
            self.model is not None
            and self.model_name == model_name
            and self.model_provider == model_provider
        ):
            print(f"Model {model_name} is already loaded. Skipping load.")
            return self.model

        print(f"Loading model: {model_name} using {model_provider}")
        self.model_provider = model_provider
        self.model_name = model_name

        if model_provider == "huggingface":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        elif model_provider == "ollama":
            self.model = ollama
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}")

        print("Model loaded successfully.")
        return self.model

    def generate_text(self, prompt, max_new_tokens=256):
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model first.")

        if self.model_provider == "huggingface":
            # Create a simple message structure
            messages = [{"role": "user", "content": prompt}]

            # Convert to model inputs
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self.model.device)

            # Set up generation config
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": 0.4,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id or 151643,  # Qwen's default EOS token
                "bos_token_id": self.tokenizer.bos_token_id,
            }

            # Handle Qwen model specifically
            if "Qwen" in self.model_name:
                generation_config.update({
                    "eos_token_id": 151643,  # Qwen's specific EOS token
                    "pad_token_id": 151643,  # Use same as EOS for Qwen
                    "top_p": 0.8,
                    "temperature": 0.4,
                    "repetition_penalty": 1.0,
                    "do_sample": False,
                    "num_return_sequences": 1,
                })

            # Generate
            outputs = self.model.generate(
                input_ids,
                **generation_config
            )

            # Decode and clean up the response
            response = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
            
            # Clean up common prefixes
            response = response.strip()
            prefixes_to_remove = [
                'Rewritten question: ',
                'Rewritten Question: ',
                'Question: ',
                '"',
                "'"
            ]
            for prefix in prefixes_to_remove:
                if response.startswith(prefix):
                    response = response[len(prefix):]
            
            # Clean up common suffixes
            suffixes_to_remove = ['"', "'", '.']
            for suffix in suffixes_to_remove:
                if response.endswith(suffix):
                    response = response[:-len(suffix)]
                    
            return response.strip()

        elif self.model_provider == "ollama":
            print(f"Generating text with {self.model_name}")
            response = self.model.chat(
                model=self.model_name,
                options={
                    "temperature": 0.4,
                    "top_p": 0.9,
                    "top_k": 50,
                    "repeat_penalty": 1.0
                },
                messages=[{"role": "user", "content": prompt}],
            )
            
            # Clean up the response the same way
            response = response["message"]["content"].strip()
            prefixes_to_remove = [
                'Rewritten question: ',
                'Rewritten Question: ',
                'Question: ',
                '"',
                "'"
            ]
            for prefix in prefixes_to_remove:
                if response.startswith(prefix):
                    response = response[len(prefix):]
            
            suffixes_to_remove = ['"', "'", '.']
            for suffix in suffixes_to_remove:
                if response.endswith(suffix):
                    response = response[:-len(suffix)]
                    
            return response.strip()

        return ""
