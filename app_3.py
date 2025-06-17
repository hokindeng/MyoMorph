#!/usr/bin/env python3
"""
MyoMorph T5 Language Model Backbone Tester
===========================================
A focused application for testing and exploring the T5 language model 
that powers MyoMorph's motion-language capabilities.
"""

import gradio as gr
import torch
import numpy as np
import os
import time
import traceback
from pathlib import Path

# MyoMorph imports
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.config import parse_args

# Configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class T5BackboneTester:
    def __init__(self):
        self.model = None
        self.datamodule = None
        self.device = None
        self.t5_model = None
        self.tokenizer = None
        
        # Initialize
        self.load_model()
    
    def load_model(self):
        """Load MyoMorph model and extract T5 backbone"""
        try:
            print("🔄 Loading MyoMorph model for T5 backbone testing...")
            
            # Parse config
            cfg = parse_args(phase="webui")
            cfg.FOLDER = 'cache'
            
            # Set device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"✅ Using GPU: {torch.cuda.get_device_name()}")
            else:
                self.device = torch.device("cpu")
                print("⚠️  Using CPU")
            
            # Build model
            self.datamodule = build_data(cfg, phase="test")
            self.model = build_model(cfg, self.datamodule)
            
            # Load checkpoint if available
            if cfg.TEST.CHECKPOINTS and os.path.exists(cfg.TEST.CHECKPOINTS):
                try:
                    state_dict = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
                    self.model.load_state_dict(state_dict)
                    print(f"✅ Loaded checkpoint: {cfg.TEST.CHECKPOINTS}")
                except Exception as e:
                    print(f"⚠️  Checkpoint loading failed: {e}")
                    print("🚀 Using randomly initialized model...")
            else:
                print("⚠️  No checkpoint found, using randomly initialized model")
            
            self.model.to(self.device)
            self.model.eval()
            
            # Extract T5 backbone
            self.t5_model = self.model.lm.language_model  # Direct T5ForConditionalGeneration
            self.tokenizer = self.model.lm.tokenizer      # Direct T5 tokenizer
            
            print("✅ T5 backbone extracted successfully!")
            print(f"🧠 Model: {type(self.t5_model).__name__}")
            print(f"📝 Tokenizer: {type(self.tokenizer).__name__}")
            print(f"🎯 Vocabulary size: {len(self.tokenizer)} tokens")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            traceback.print_exc()
            raise e
    
    def get_model_stats(self):
        """Get detailed T5 model statistics"""
        if not self.t5_model:
            return "❌ Model not loaded"
        
        # Count parameters
        total_params = sum(p.numel() for p in self.t5_model.parameters())
        trainable_params = sum(p.numel() for p in self.t5_model.parameters() if p.requires_grad)
        
        # Get model config
        config = self.t5_model.config
        
        stats = f"""
🧠 **T5 Language Model Backbone Statistics**

**Architecture Details:**
• Model: {type(self.t5_model).__name__}
• Parameters: {total_params:,} total ({trainable_params:,} trainable)
• Model Size: ~{total_params / 1e6:.1f}M parameters
• Device: {self.t5_model.device}

**T5 Configuration:**
• Model Type: {config.model_type}
• Hidden Size: {config.d_model}
• Number of Layers: {config.num_layers} (encoder) + {config.num_decoder_layers} (decoder)
• Attention Heads: {config.num_heads}
• Feed Forward Size: {config.d_ff}
• Vocabulary Size: {config.vocab_size}

**MyoMorph Extensions:**
• Total Vocabulary: {len(self.tokenizer)} tokens
• Motion Tokens: {self.model.lm.m_codebook_size} motion IDs
• Special Tokens: 3 (start, end, mask)
• Max Length: {self.model.lm.max_length} tokens

**Memory Usage:**
• Model Memory: ~{total_params * 4 / 1e6:.1f} MB (FP32)
• Current Device: {next(self.t5_model.parameters()).device}
"""
        return stats
    
    def test_generation(self, prompt, max_length=100, temperature=0.7, top_p=0.9, num_beams=1, do_sample=True):
        """Test T5 text generation capabilities"""
        try:
            if not self.t5_model:
                return "❌ Model not loaded"
            
            print(f"🔄 Testing T5 generation: '{prompt}'")
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                max_length=self.model.lm.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate
            start_time = time.time()
            with torch.no_grad():
                outputs = self.t5_model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            generation_time = time.time() - start_time
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Stats
            input_tokens = len(inputs.input_ids[0])
            output_tokens = len(outputs[0])
            generated_tokens = output_tokens - input_tokens
            
            result = f"""
🤖 **T5 Generation Test Results**

**Input:** {prompt}
**Generated:** {generated_text}

**Performance:**
• Generation Time: {generation_time:.2f} seconds
• Tokens/Second: {generated_tokens/generation_time:.1f}
• Input Tokens: {input_tokens}
• Generated Tokens: {generated_tokens}
• Total Tokens: {output_tokens}

**Parameters Used:**
• Max Length: {max_length}
• Temperature: {temperature}
• Top-p: {top_p}
• Beams: {num_beams}
• Sampling: {do_sample}
"""
            return result
            
        except Exception as e:
            return f"❌ Generation failed: {str(e)}"
    
    def analyze_tokens(self, text):
        """Analyze tokenization including motion tokens"""
        try:
            if not self.tokenizer:
                return "❌ Tokenizer not loaded"
            
            # Tokenize
            tokens = self.tokenizer.encode(text, return_tensors="pt")[0]
            token_texts = self.tokenizer.convert_ids_to_tokens(tokens)
            
            # Categorize tokens
            motion_tokens = []
            special_tokens = []
            regular_tokens = []
            
            for i, (token_id, token_text) in enumerate(zip(tokens, token_texts)):
                if 'motion_id' in token_text:
                    motion_tokens.append((i, int(token_id), token_text))
                elif token_text.startswith('<') and token_text.endswith('>'):
                    special_tokens.append((i, int(token_id), token_text))
                else:
                    regular_tokens.append((i, int(token_id), token_text))
            
            analysis = f"""
🔍 **Token Analysis for:** "{text}"

**Token Statistics:**
• Total Tokens: {len(tokens)}
• Regular Tokens: {len(regular_tokens)}
• Motion Tokens: {len(motion_tokens)}
• Special Tokens: {len(special_tokens)}

**Token Breakdown:**
"""
            
            # Show all tokens
            for i, (token_id, token_text) in enumerate(zip(tokens, token_texts)):
                if 'motion_id' in token_text:
                    token_type = "🎭"
                elif token_text.startswith('<'):
                    token_type = "⚙️"
                else:
                    token_type = "📝"
                
                analysis += f"{token_type} {i:2d}: `{token_text:<15}` (ID: {int(token_id):5d})\n"
            
            if motion_tokens:
                analysis += f"\n**Motion Tokens Found:**\n"
                for pos, token_id, token_text in motion_tokens:
                    motion_id = token_text.split('_')[-1].replace('>', '')
                    analysis += f"🎭 Position {pos}: {token_text} → Motion ID {motion_id}\n"
            
            return analysis
            
        except Exception as e:
            return f"❌ Token analysis failed: {str(e)}"
    
    def test_motion_integration(self):
        """Test motion token integration"""
        try:
            motion_examples = [
                "<motion_id_0><motion_id_1><motion_id_2>",
                "Generate motion: <motion_id_512>walking<motion_id_513>",
                "A person <motion_id_0> walks forward <motion_id_1>",
                f"<motion_id_{self.model.lm.m_codebook_size}>motion sequence<motion_id_{self.model.lm.m_codebook_size+1}>"
            ]
            
            results = "🎭 **Motion Token Integration Test**\n\n"
            
            for i, example in enumerate(motion_examples):
                results += f"**Test {i+1}:** {example}\n"
                analysis = self.analyze_tokens(example)
                # Extract just the token count summary
                lines = analysis.split('\n')
                for line in lines:
                    if 'Total Tokens:' in line or 'Motion Tokens:' in line:
                        results += f"  {line.strip()}\n"
                results += "\n"
            
            # Test vocabulary boundaries
            results += "**Vocabulary Boundaries:**\n"
            results += f"• Standard T5 vocab: 0 - {32099}\n"
            results += f"• Motion tokens: {32100} - {32100 + self.model.lm.m_codebook_size - 1}\n"
            results += f"• Special tokens: {32100 + self.model.lm.m_codebook_size} - {len(self.tokenizer) - 1}\n"
            
            return results
            
        except Exception as e:
            return f"❌ Motion integration test failed: {str(e)}"

# Initialize the tester
print("🚀 Initializing T5 Backbone Tester...")
tester = T5BackboneTester()

# Interface functions
def get_stats_interface():
    return tester.get_model_stats()

def test_generation_interface(prompt, max_length, temperature, top_p, num_beams, do_sample):
    return tester.test_generation(prompt, max_length, temperature, top_p, num_beams, do_sample)

def analyze_tokens_interface(text):
    return tester.analyze_tokens(text)

def test_motion_interface():
    return tester.test_motion_integration()

# Create Gradio interface
with gr.Blocks(title="MyoMorph T5 Backbone Tester", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 MyoMorph T5 Language Model Backbone Tester")
    gr.Markdown("Explore and test the T5-Base language model that powers MyoMorph's motion-language capabilities.")
    
    with gr.Tab("📊 Model Information"):
        gr.Markdown("### T5 Model Statistics and Configuration")
        
        stats_button = gr.Button("📋 Get Model Statistics", variant="primary", size="lg")
        stats_output = gr.Textbox(label="Model Statistics", lines=25, max_lines=25)
        
        stats_button.click(get_stats_interface, outputs=[stats_output])
    
    with gr.Tab("🚀 Generation Testing"):
        gr.Markdown("### Test T5 Text Generation Capabilities")
        
        with gr.Row():
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(
                    label="Input Prompt",
                    placeholder="Enter text for T5 to generate from...",
                    lines=3
                )
                
                with gr.Row():
                    max_length = gr.Slider(20, 200, 100, label="Max Length")
                    temperature = gr.Slider(0.1, 2.0, 0.7, label="Temperature")
                
                with gr.Row():
                    top_p = gr.Slider(0.1, 1.0, 0.9, label="Top-p")
                    num_beams = gr.Slider(1, 5, 1, label="Beams")
                    do_sample = gr.Checkbox(True, label="Sampling")
                
                generate_button = gr.Button("🚀 Generate", variant="primary")
                
            with gr.Column(scale=1):
                gr.Markdown("### Example Prompts:")
                example_prompts = [
                    "Complete this sentence: A person walks",
                    "Describe human motion:",
                    "Generate motion: dancing",
                    "The athlete performs",
                    "Human movement can be described as",
                    "Walking is characterized by"
                ]
                
                for prompt in example_prompts:
                    gr.Button(prompt, size="sm").click(
                        lambda x=prompt: x, outputs=prompt_input
                    )
        
        generation_output = gr.Textbox(label="Generation Results", lines=15)
        
        generate_button.click(
            test_generation_interface,
            inputs=[prompt_input, max_length, temperature, top_p, num_beams, do_sample],
            outputs=[generation_output]
        )
    
    with gr.Tab("🔍 Token Analysis"):
        gr.Markdown("### Analyze Text Tokenization (Including Motion Tokens)")
        
        with gr.Row():
            with gr.Column(scale=2):
                token_input = gr.Textbox(
                    label="Text to Analyze",
                    placeholder="Enter text to see how T5 tokenizes it...",
                    lines=3
                )
                analyze_button = gr.Button("🔍 Analyze Tokens", variant="primary")
                
            with gr.Column(scale=1):
                gr.Markdown("### Test Examples:")
                token_examples = [
                    "Hello world",
                    "A person walks forward",
                    "<motion_id_0><motion_id_1>",
                    "Generate motion: walking",
                    "Motion tokens: <motion_id_512>",
                    "🎭 Human movement analysis"
                ]
                
                for example in token_examples:
                    gr.Button(example, size="sm").click(
                        lambda x=example: x, outputs=token_input
                    )
        
        token_output = gr.Textbox(label="Token Analysis Results", lines=20)
        
        analyze_button.click(
            analyze_tokens_interface,
            inputs=[token_input],
            outputs=[token_output]
        )
    
    with gr.Tab("🎭 Motion Integration"):
        gr.Markdown("### Test Motion Token Integration")
        gr.Markdown("Verify how motion tokens are integrated into the T5 vocabulary.")
        
        motion_button = gr.Button("🎭 Test Motion Integration", variant="primary", size="lg")
        motion_output = gr.Textbox(label="Motion Integration Test Results", lines=20)
        
        motion_button.click(test_motion_interface, outputs=[motion_output])
    
    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
        ## About This Tester
        
        This application provides direct access to the T5-Base language model that powers MyoMorph.
        
        ### What You Can Test:
        
        **📊 Model Information:**
        - Detailed T5 architecture specifications
        - Parameter counts and memory usage
        - MyoMorph-specific extensions
        
        **🚀 Generation Testing:**
        - Direct T5 text generation
        - Adjustable generation parameters
        - Performance metrics
        
        **🔍 Token Analysis:**
        - See exactly how text gets tokenized
        - Identify motion tokens vs regular tokens
        - Understand vocabulary structure
        
        **🎭 Motion Integration:**
        - Test motion token functionality
        - Verify vocabulary boundaries
        - Explore motion-text integration
        
        ### Technical Details:
        - **Base Model:** T5-Base (770M parameters)
        - **Extended Vocabulary:** 32,100 + 515 tokens
        - **Motion Tokens:** 512 discrete motion IDs
        - **Architecture:** Encoder-Decoder Transformer
        
        ### Usage Tips:
        - Start with "Model Information" to understand the architecture
        - Use "Generation Testing" to see T5's language capabilities
        - Try "Token Analysis" to understand motion-text integration
        - Check "Motion Integration" to verify motion token functionality
        """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=8891,
        debug=True,
        share=True
    ) 