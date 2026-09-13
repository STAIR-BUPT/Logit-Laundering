import time
import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer

import nltk
from nltk.tokenize import sent_tokenize

class DipperParaphraser(object):
    def __init__(self, model="",  # set via --model or $MODEL_ROOT/dipper-paraphraser-xxl
                 verbose=True, device_map=None):
        """
        Args:
            model: Model path (default: ''; set via argument or $MODEL_ROOT env var).
            verbose: Whether to print loading info.
            device_map: Device mapping. Accepts "auto", a device string (e.g. "cuda:0"),
                        or a dict. If None, auto-detected from CUDA_VISIBLE_DEVICES.
        """
        if verbose:
            print("Initializing DipperParaphraser model...")

        # Download NLTK data if not already present
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
        except Exception as e:
            if verbose:
                print(f"NLTK data download warning: {e}; continuing...")

        time1 = time.time()
        if verbose:
            print("Loading T5 tokenizer...")
        self.tokenizer = T5Tokenizer.from_pretrained(
            os.environ.get('T5_MODEL_PATH', '/t5-v1_1-xxl')
        )

        # Auto-detect device_map from CUDA_VISIBLE_DEVICES if not specified
        if device_map is None:
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            if cuda_visible:
                # CUDA_VISIBLE_DEVICES restricts visible devices; "auto" will use them
                device_map = "auto"
                if verbose:
                    print(f"Detected CUDA_VISIBLE_DEVICES={cuda_visible}; using those devices.")
            else:
                # No restriction; use all available devices
                device_map = "auto"
                if verbose:
                    print("CUDA_VISIBLE_DEVICES not set; using all available devices.")

        if verbose:
            print(f"Loading DIPPER model: {model} (device_map={device_map})...")
            print("Note: the model is large and may take several minutes to load...")

        self.model = T5ForConditionalGeneration.from_pretrained(
            model,
            device_map=device_map,
            torch_dtype="auto",
            load_in_4bit=True,
        )
        if verbose:
            load_time = time.time() - time1
            print(f"✅ {model} loaded in {load_time:.2f}s")
            # Print actual device placement
            if hasattr(self.model, 'hf_device_map'):
                print(f"Model device map: {self.model.hf_device_map}")
            elif hasattr(self.model, 'device'):
                print(f"Model device: {self.model.device}")

        # If device_map is not "auto", move the model to the specified device manually
        # (when device_map="auto", the model is already placed automatically)
        if device_map != "auto" and not isinstance(device_map, dict):
            # Single device string — move model there
            if isinstance(device_map, str) and device_map.startswith('cuda'):
                self.model = self.model.to(device_map)
        
        self.model.eval()

    def paraphrase(self, input_text, lex_diversity, order_diversity, prefix="", sent_interval=3, **kwargs):
        """Paraphrase a text using the DIPPER model.

        Args:
            input_text (str): The text to paraphrase. Make sure to mark the sentence to be paraphrased between <sent> and </sent> blocks, keeping space on either side.
            lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            **kwargs: Additional keyword arguments like top_p, top_k, max_length.
        """
        assert lex_diversity in [0, 20, 40, 60, 80, 100], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
        assert order_diversity in [0, 20, 40, 60, 80, 100], "Order diversity must be one of 0, 20, 40, 60, 80, 100."

        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)

        input_text = " ".join(input_text.split())
        sentences = sent_tokenize(input_text)
        prefix = " ".join(prefix.replace("\n", " ").split())
        output_text = ""
        outputs_list=[]

        for sent_idx in range(0, len(sentences), sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + sent_interval])
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"

            final_input = self.tokenizer([final_input_text], return_tensors="pt")
            
            # Determine which device to place the input on.
            # When device_map="auto", the model may span multiple devices;
            # input should go on the first device (where the first layer resides).
            if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
                # Get the first device
                first_device = list(self.model.hf_device_map.values())[0]
                if isinstance(first_device, (list, tuple)):
                    first_device = first_device[0]
                final_input = {k: v.to(first_device) for k, v in final_input.items()}
            elif hasattr(self.model, 'device'):
                final_input = {k: v.to(self.model.device) for k, v in final_input.items()}
            else:
                # Fall back to cuda:0
                final_input = {k: v.cuda() for k, v in final_input.items()}

            with torch.inference_mode():
                outputs = self.model.generate(**final_input, **kwargs)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            prefix += " " + outputs[0]
            output_text += " " + outputs[0]
            outputs_list.append(outputs[0])

        return output_text

class Authormist:
    def __init__(self):
        self.device='cuda' if torch.cuda.is_available() else "cpu"
        self.model_name = "authormist/authormist-originality"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                torch_dtype="auto",).to(self.device)

    def paraphrase(self, input_text):
        # Prepare input text
        ai_text = "Your AI-generated text here..."
        prompt = f"""Please paraphrase the following text to make it more human-like while preserving the original meaning:

        {input_text}

        Paraphrased text:"""

        # Generate paraphrased text
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs.input_ids.to(self.device),
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        paraphrased_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_text = paraphrased_text.split("Paraphrased text:")[1].strip()
        return output_text

if __name__ == "__main__":
    # dp = DipperParaphraser()

    # prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote valley."
    input_text = "They have never been known to mingle with humans. Today, it is believed these unicorns live in an unspoilt environment which is surrounded by mountains. Its edge is protected by a thick wattle of wattle trees, giving it a majestic appearance. Along with their so-called miracle of multicolored coat, their golden coloured feather makes them look like mirages. Some of them are rumored to be capable of speaking a large amount of different languages. They feed on elk and goats as they were selected from those animals that possess a fierceness to them, and can \"eat\" them with their long horns."

    # print(f"Input = {prompt} <sent> {input_text} </sent>\n")
    # output_l60_sample = dp.paraphrase(input_text, lex_diversity=60, order_diversity=0, prefix=prompt, do_sample=True, top_p=0.75, top_k=None, max_length=512)
    # print(f"Output (Lexical diversity = 60, Sample p = 0.75) = {output_l60_sample}\n")

    am=Authormist()
    output_text=am.paraphrase(input_text)
