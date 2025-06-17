import gradio as gr
import torch
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import traceback

# MotionGPT imports
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.config import parse_args
import mGPT.render.matplot.plot_3d_global as plot_3d

# Configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Set GPU

class MotionGPTApp:
    def __init__(self):
        self.model = None
        self.datamodule = None
        self.device = None
        self.output_dir = Path("cache")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize the MotionGPT model"""
        try:
            print("🔄 Loading MyoMorph model...")
            
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
            
            # Build data module and model
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
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Model initialization failed: {e}")
            traceback.print_exc()
            raise e
    
    def text_to_motion(self, text_prompt, motion_length=120):
        """Generate motion from text"""
        try:
            if not self.model:
                return "❌ Model not initialized", None, None
            
            if not text_prompt.strip():
                return "❌ Please enter a text prompt", None, None
            
            print(f"🔄 Generating motion for: '{text_prompt}'")
            
            # Prepare batch
            batch = {
                "text": [text_prompt],
                "length": [motion_length]
            }
            
            # Generate motion
            with torch.no_grad():
                outputs = self.model(batch, task="t2m")
            
            # Extract results
            generated_motion = outputs["feats"][0].cpu().numpy()  # (seq_len, feat_dim)
            generated_joints = outputs["joints"][0].cpu().numpy()  # (seq_len, num_joints, 3)
            output_text = outputs.get("texts", [text_prompt])[0]
            
            # Save motion data
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            motion_file = self.output_dir / f"motion_{timestamp}.npy"
            np.save(motion_file, generated_joints)
            
            # Create simple visualization
            video_file = self.create_motion_video(generated_joints, timestamp)
            
            result_text = f"✅ Motion generated successfully!\n"
            result_text += f"📝 Input: {text_prompt}\n"
            result_text += f"🎬 Motion length: {len(generated_joints)} frames\n"
            result_text += f"💾 Saved to: {motion_file.name}"
            
            return result_text, str(video_file), str(motion_file)
            
        except Exception as e:
            error_msg = f"❌ Text-to-motion generation failed: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return error_msg, None, None
    
    def motion_to_text(self, motion_file):
        """Generate text description from motion file"""
        try:
            if not self.model:
                return "❌ Model not initialized"
            
            if not motion_file:
                return "❌ Please upload a motion file"
            
            print(f"🔄 Generating text for motion file: {motion_file}")
            
            # Load motion data
            if isinstance(motion_file, str):
                motion_data = np.load(motion_file)
            else:
                motion_data = np.load(motion_file.name)
            
            # Convert to tensor and normalize
            if len(motion_data.shape) == 2:
                motion_data = motion_data[None]  # Add batch dimension
            
            motion_tensor = torch.tensor(motion_data, device=self.device, dtype=torch.float32)
            
            # Encode motion to tokens
            motion_tokens = []
            lengths = []
            
            for i in range(len(motion_tensor)):
                # Normalize motion if needed
                feats = motion_tensor[i:i+1]
                if hasattr(self.datamodule, 'normalize'):
                    feats = self.datamodule.normalize(feats)
                
                # Encode to motion tokens
                motion_token, _ = self.model.vae.encode(feats)
                motion_tokens.append(motion_token[0])
                lengths.append(motion_token.shape[1])
            
            # Generate text description
            with torch.no_grad():
                generated_texts = self.model.lm.generate_conditional(
                    motion_tokens=motion_tokens,
                    lengths=lengths,
                    task="m2t",
                    stage='test'
                )
            
            result_text = f"✅ Text generated successfully!\n"
            result_text += f"🎬 Motion length: {motion_data.shape[0]} frames\n"
            result_text += f"📝 Description: {generated_texts[0]}"
            
            return result_text
            
        except Exception as e:
            error_msg = f"❌ Motion-to-text generation failed: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return error_msg
    
    def text_to_text_chat(self, user_input):
        """Generate text responses for motion-related conversations"""
        try:
            if not self.model:
                return "❌ Model not initialized"
            
            if not user_input.strip():
                return "❌ Please enter a message"
            
            print(f"🔄 Generating response for: '{user_input}'")
            
            # Handle conversational inputs first
            conversational_response = self.handle_conversational_input(user_input)
            if conversational_response:
                return conversational_response
            
            # Enhance the prompt to be more motion-focused
            enhanced_prompt = self.enhance_chat_prompt(user_input)
            
            # Use the language model directly for text generation
            with torch.no_grad():
                # Generate response using the direct generation method
                # Note: generate_direct returns (tokens, text) - we want the text
                _, response_texts = self.model.lm.generate_direct(
                    [enhanced_prompt],
                    max_length=150,
                    num_beams=2,
                    do_sample=True
                )
            
            if not response_texts or not response_texts[0]:
                return "🤖 MyoMorph: I couldn't generate a response. Try asking about specific human movements or motions!"
            
            response = response_texts[0]
            
            # Clean up the response 
            response = self.clean_chat_response(response, user_input, enhanced_prompt)
            
            result_text = f"🤖 MyoMorph: {response}"
            
            return result_text
            
        except Exception as e:
            error_msg = f"❌ Chat failed: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return f"🤖 MyoMorph: I'm having trouble processing that. Try asking about specific human movements like walking, running, or dancing!"
    
    def handle_conversational_input(self, user_input):
        """Handle greetings and non-motion conversational inputs"""
        user_lower = user_input.lower().strip()
        
        # Greetings
        if any(greeting in user_lower for greeting in ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']):
            return "🤖 MyoMorph: Hello! I'm MyoMorph, your motion and movement assistant. I can help you understand human motion, generate movement descriptions, or create motions from text. What kind of movement would you like to explore today?"
        
        # Farewells
        if any(farewell in user_lower for farewell in ['bye', 'goodbye', 'see you', 'farewell', 'later']):
            return "🤖 MyoMorph: Goodbye! Feel free to come back if you want to explore more about human motion and movement!"
        
        # Thank you
        if any(thanks in user_lower for thanks in ['thank', 'thanks', 'thx']):
            return "🤖 MyoMorph: You're welcome! Happy to help you understand human motion and movement!"
        
        # Who are you / What are you
        if any(question in user_lower for question in ['who are you', 'what are you', 'what do you do', 'tell me about yourself']):
            return "🤖 MyoMorph: I'm MyoMorph, a specialized AI assistant for human motion and movement. I can generate realistic human motions from text descriptions, describe motions in natural language, and answer questions about how people move. Try asking me about walking, dancing, sports, or any human movement!"
        
        # Help requests
        if any(help_word in user_lower for help_word in ['help', 'how do i', 'what can you', 'capabilities']):
            return "🤖 MyoMorph: I can help you with:\n• Generate human motion from descriptions (Text-to-Motion tab)\n• Describe motions from data files (Motion-to-Text tab)\n• Answer questions about human movement and body mechanics\n• Explain how different activities look when performed\n\nTry asking: 'How do people walk?' or 'What does running look like?'"
        
        # Random/unclear inputs
        if len(user_lower) < 3 or user_lower in ['test', 'testing', '123', 'abc']:
            return "🤖 MyoMorph: I'm here to help with human motion and movement! Try asking me something like 'How do people dance?' or 'What does jumping look like?'"
        
        return None  # Not a conversational input, proceed with motion processing
    
    def enhance_chat_prompt(self, user_input):
        """Enhance user input to be more motion-focused for better model performance"""
        # Check if the input is already motion-related
        motion_keywords = ['walk', 'run', 'dance', 'jump', 'move', 'motion', 'gesture', 'pose', 'stretch', 'exercise', 'sport', 'arm', 'leg', 'body', 'human', 'person', 'kick', 'punch', 'bend', 'twist', 'turn', 'step', 'hop', 'skip', 'crawl', 'climb', 'swim', 'throw', 'catch']
        
        if any(keyword in user_input.lower() for keyword in motion_keywords):
            # Already motion-related, just add context
            return f"Describe the motion: {user_input}"
        else:
            # For non-motion topics, try to guide toward motion
            return f"Explain how humans move when: {user_input}"
    
    def clean_chat_response(self, response, original_input, enhanced_prompt):
        """Clean up the generated response"""
        if not response:
            return "I need more information to help you with that motion."
        
        # Remove the enhanced prompt if it appears in the response
        if enhanced_prompt in response:
            response = response.replace(enhanced_prompt, "").strip()
        
        # Remove common prefixes that might appear
        prefixes_to_remove = ["Describe motion:", "Generate motion:", "Motion:", "Human motion:"]
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # If response is too short or empty, provide a helpful message
        if len(response.strip()) < 10:
            return "I can help you understand human motion and movement. Try asking about specific activities like walking, dancing, or sports movements!"
        
        return response.strip()
    
    def create_motion_video(self, joints, timestamp):
        """Create a simple motion visualization"""
        try:
            video_file = self.output_dir / f"motion_{timestamp}.mp4"
            
            # Use the existing plot_3d module for visualization
            if len(joints.shape) == 3:
                joints = joints[None]  # Add batch dimension
            
            # Create GIF first
            gif_file = self.output_dir / f"motion_{timestamp}.gif"
            plot_3d.draw_to_batch(joints, [''], [str(gif_file)])
            
            # Convert to MP4 if possible
            try:
                import moviepy.editor as mp
                clip = mp.VideoFileClip(str(gif_file))
                clip.write_videofile(str(video_file), verbose=False, logger=None)
                clip.close()
                return video_file
            except:
                # Return GIF if MP4 conversion fails
                return gif_file
                
        except Exception as e:
            print(f"⚠️  Video creation failed: {e}")
            return None
    
    def get_language_model_info(self):
        """Get information about the underlying language model"""
        if not self.model:
            return "❌ Model not initialized"
        
        lm = self.model.lm
        info = f"🧠 **Language Model Backbone Information:**\n\n"
        info += f"**Model Type:** {lm.lm_type} (encoder-decoder)\n"
        info += f"**Architecture:** {type(lm.language_model).__name__}\n"
        info += f"**Vocabulary Size:** {len(lm.tokenizer)} tokens\n"
        info += f"**Motion Codebook Size:** {lm.m_codebook_size} motion tokens\n"
        info += f"**Max Length:** {lm.max_length} tokens\n"
        info += f"**Device:** {lm.language_model.device}\n"
        info += f"**Parameters:** ~770M (T5-Base backbone)\n\n"
        info += f"**Special Tokens:**\n"
        info += f"• Motion tokens: `<motion_id_0>` to `<motion_id_{lm.m_codebook_size-1}>`\n"
        info += f"• Start token: `<motion_id_{lm.m_codebook_size}>`\n"
        info += f"• End token: `<motion_id_{lm.m_codebook_size+1}>`\n"
        info += f"• Mask token: `<motion_id_{lm.m_codebook_size+2}>`\n"
        
        return info
    
    def use_t5_directly(self, input_text, task_type="generate", max_length=100):
        """Directly use the T5 language model backbone"""
        try:
            if not self.model:
                return "❌ Model not initialized"
            
            if not input_text.strip():
                return "❌ Please enter input text"
            
            # Access the T5 model directly
            t5_model = self.model.lm.language_model
            tokenizer = self.model.lm.tokenizer
            
            print(f"🔄 Using T5 directly for: '{input_text}'")
            
            # Prepare input based on task type
            if task_type == "generate":
                # Direct text generation
                prompt = input_text
            elif task_type == "complete":
                # Text completion
                prompt = f"Complete this text: {input_text}"
            elif task_type == "question":
                # Question answering
                prompt = f"Answer this question: {input_text}"
            elif task_type == "summarize":
                # Summarization
                prompt = f"Summarize: {input_text}"
            elif task_type == "translate":
                # Translation (example)
                prompt = f"Translate to simple language: {input_text}"
            else:
                prompt = input_text
            
            # Tokenize input
            inputs = tokenizer(
                prompt,
                max_length=self.model.lm.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate using T5 directly
            with torch.no_grad():
                outputs = t5_model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=max_length,
                    num_beams=3,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up output
            if prompt in generated_text:
                generated_text = generated_text.replace(prompt, "").strip()
            
            result = f"🤖 **T5 Backbone Output:**\n"
            result += f"**Task:** {task_type.title()}\n"
            result += f"**Input:** {input_text}\n"
            result += f"**Generated:** {generated_text}\n\n"
            result += f"**Model Details:**\n"
            result += f"• Tokens generated: {len(outputs[0]) - len(inputs.input_ids[0])}\n"
            result += f"• Input length: {len(inputs.input_ids[0])} tokens\n"
            result += f"• Output length: {len(outputs[0])} tokens"
            
            return result
            
        except Exception as e:
            error_msg = f"❌ T5 direct usage failed: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return error_msg
    
    def analyze_motion_tokens(self, text):
        """Analyze how text gets converted to motion tokens"""
        try:
            if not self.model:
                return "❌ Model not initialized"
            
            tokenizer = self.model.lm.tokenizer
            
            # Tokenize the text
            tokens = tokenizer.encode(text)
            token_texts = tokenizer.convert_ids_to_tokens(tokens)
            
            # Find motion tokens
            motion_tokens = []
            regular_tokens = []
            
            for i, (token_id, token_text) in enumerate(zip(tokens, token_texts)):
                if 'motion_id' in token_text:
                    motion_tokens.append((i, token_id, token_text))
                else:
                    regular_tokens.append((i, token_id, token_text))
            
            result = f"🔍 **Token Analysis for:** '{text}'\n\n"
            result += f"**Total Tokens:** {len(tokens)}\n"
            result += f"**Regular Tokens:** {len(regular_tokens)}\n"
            result += f"**Motion Tokens:** {len(motion_tokens)}\n\n"
            
            result += f"**Token Breakdown:**\n"
            for i, (pos, token_id, token_text) in enumerate(zip(range(len(tokens)), tokens, token_texts)):
                if i < 20:  # Show first 20 tokens
                    token_type = "🎭" if 'motion_id' in token_text else "📝"
                    result += f"{token_type} {pos}: `{token_text}` (ID: {token_id})\n"
                elif i == 20 and len(tokens) > 20:
                    result += f"... and {len(tokens) - 20} more tokens\n"
            
            if motion_tokens:
                result += f"\n**Motion Tokens Found:**\n"
                for pos, token_id, token_text in motion_tokens[:10]:
                    result += f"🎭 Position {pos}: `{token_text}` (ID: {token_id})\n"
            
            return result
            
        except Exception as e:
            return f"❌ Token analysis failed: {str(e)}"

# Initialize the app
app = MotionGPTApp()

def t2m_interface(text_prompt):
    """Interface function for text-to-motion"""
    result_text, video_file, motion_file = app.text_to_motion(text_prompt)
    return result_text, video_file, motion_file

def m2t_interface(motion_file):
    """Interface function for motion-to-text"""
    result_text = app.motion_to_text(motion_file)
    return result_text

def t2t_interface(user_input):
    """Interface function for text-to-text chat"""
    result_text = app.text_to_text_chat(user_input)
    return result_text

def t5_direct_interface(input_text, task_type, max_length):
    """Interface function for direct T5 usage"""
    result_text = app.use_t5_directly(input_text, task_type, max_length)
    return result_text

def get_model_info_interface():
    """Interface function to get language model info"""
    return app.get_language_model_info()

def analyze_tokens_interface(text):
    """Interface function for token analysis"""
    return app.analyze_motion_tokens(text)

def create_sample_motion():
    """Create a sample motion for testing"""
    try:
        # Create simple walking motion (dummy data)
        frames = 60
        joints = 22
        motion = np.random.randn(frames, joints, 3) * 0.1
        
        # Add some walking-like motion
        for i in range(frames):
            t = i / frames * 2 * np.pi
            motion[i, 0, 0] = i * 0.02  # Forward movement
            motion[i, 0, 1] = np.sin(t * 4) * 0.05  # Vertical bobbing
        
        sample_file = app.output_dir / "sample_motion.npy"
        np.save(sample_file, motion)
        return str(sample_file)
    except Exception as e:
        print(f"Sample creation failed: {e}")
        return None

# Create Gradio interface
with gr.Blocks(title="MyoMorph - Simplified Interface") as demo:
    gr.Markdown("# 🤖 MyoMorph - Simplified Interface")
    gr.Markdown("Generate human motion from text descriptions or describe motions in natural language.")
    
    with gr.Tab("💬 Text-to-Motion"):
        gr.Markdown("### Generate human motion from text descriptions")
        
        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Text Description",
                    placeholder="Enter a description of human motion (e.g., 'A person walks forward and waves')",
                    lines=3
                )
                t2m_button = gr.Button("🎬 Generate Motion", variant="primary")
            
            with gr.Column(scale=1):
                gr.Markdown("### Examples:")
                example_texts = [
                    "A person walks forward slowly",
                    "Someone is dancing with their arms up",
                    "A person sits down and then stands up",
                    "Someone is running and then stops",
                    "A person waves hello with their right hand"
                ]
                for example in example_texts:
                    gr.Button(example, size="sm").click(
                        lambda x=example: x, outputs=text_input
                    )
        
        with gr.Row():
            t2m_output = gr.Textbox(label="Generation Result", lines=4)
        
        with gr.Row():
            video_output = gr.Video(label="Generated Motion Video")
            motion_file_output = gr.File(label="Motion Data (.npy)")
        
        t2m_button.click(
            t2m_interface,
            inputs=[text_input],
            outputs=[t2m_output, video_output, motion_file_output]
        )
    
    with gr.Tab("🎭 Motion-to-Text"):
        gr.Markdown("### Generate text descriptions from motion data")
        
        with gr.Row():
            with gr.Column():
                motion_input = gr.File(
                    label="Upload Motion File (.npy)",
                    file_types=[".npy"]
                )
                
                with gr.Row():
                    m2t_button = gr.Button("📝 Generate Text", variant="primary")
                    sample_button = gr.Button("📄 Use Sample Motion", variant="secondary")
                
                m2t_output = gr.Textbox(label="Generated Description", lines=6)
        
        m2t_button.click(
            m2t_interface,
            inputs=[motion_input],
            outputs=[m2t_output]
        )
        
        sample_button.click(
            create_sample_motion,
            outputs=[motion_input]
        )
    
    with gr.Tab("💬 Chat"):
        gr.Markdown("### Chat with MyoMorph about motion and movement")
        gr.Markdown("💡 **MyoMorph specializes in human motion!** I can handle basic conversation but excel at movement questions.")
        
        with gr.Row():
            with gr.Column(scale=3):
                chat_input = gr.Textbox(
                    label="Your Message",
                    placeholder="Try: 'Hello!' or 'How do people walk?' or 'What does dancing look like?'",
                    lines=2
                )
                chat_button = gr.Button("💬 Send Message", variant="primary")
                
                chat_output = gr.Textbox(label="MyoMorph Response", lines=8)
            
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Conversation Examples:")
                gr.Markdown("• Hello!\n• What can you do?\n• Help me understand")
                
                gr.Markdown("### 🏃 Motion Questions:")
                example_questions = [
                    "How do people walk?",
                    "What does running look like?",
                    "Describe dance movements",
                    "How do you jump properly?",
                    "What happens when someone waves?",
                    "Describe sitting down motion",
                    "How do people stretch their arms?",
                    "What are martial arts kicks like?"
                ]
                
                for example in example_questions:
                    gr.Button(example, size="sm").click(
                        lambda x=example: x, outputs=chat_input
                    )
        
        chat_button.click(
            t2t_interface,
            inputs=[chat_input],
            outputs=[chat_output]
        )
        
        # Clear input after sending
        chat_button.click(
            lambda: "", 
            outputs=[chat_input]
        )
    
    with gr.Tab("🧠 T5 Backbone"):
        gr.Markdown("### Direct Access to T5 Language Model")
        gr.Markdown("💡 **Explore the underlying T5-Base language model that powers MyoMorph!**")
        
        with gr.Row():
            with gr.Column():
                # Model Information Section
                gr.Markdown("#### 📊 Model Information")
                info_button = gr.Button("📋 Get Model Info", variant="secondary")
                model_info_output = gr.Textbox(label="Language Model Information", lines=15)
                
                info_button.click(
                    get_model_info_interface,
                    outputs=[model_info_output]
                )
        
        with gr.Row():
            with gr.Column(scale=2):
                # Direct T5 Usage Section
                gr.Markdown("#### 🤖 Direct T5 Generation")
                
                t5_input = gr.Textbox(
                    label="Input Text",
                    placeholder="Enter text for T5 to process...",
                    lines=3
                )
                
                with gr.Row():
                    task_type = gr.Dropdown(
                        choices=["generate", "complete", "question", "summarize", "translate"],
                        value="generate",
                        label="Task Type"
                    )
                    max_length = gr.Slider(
                        minimum=20,
                        maximum=200,
                        value=100,
                        step=10,
                        label="Max Length"
                    )
                
                t5_button = gr.Button("🚀 Use T5 Directly", variant="primary")
                t5_output = gr.Textbox(label="T5 Direct Output", lines=10)
                
                t5_button.click(
                    t5_direct_interface,
                    inputs=[t5_input, task_type, max_length],
                    outputs=[t5_output]
                )
            
            with gr.Column(scale=1):
                # Token Analysis Section
                gr.Markdown("#### 🔍 Token Analysis")
                
                token_input = gr.Textbox(
                    label="Text to Analyze",
                    placeholder="Enter text to see how it's tokenized...",
                    lines=2
                )
                
                analyze_button = gr.Button("🔍 Analyze Tokens", variant="secondary")
                token_output = gr.Textbox(label="Token Analysis", lines=12)
                
                analyze_button.click(
                    analyze_tokens_interface,
                    inputs=[token_input],
                    outputs=[token_output]
                )
                
                # Example inputs
                gr.Markdown("##### Examples:")
                example_inputs = [
                    "A person walks forward",
                    "Generate motion: dancing",
                    "<motion_id_0><motion_id_1>",
                    "Complete this sentence: The athlete",
                    "What is the capital of France?"
                ]
                
                for example in example_inputs:
                    gr.Button(example, size="sm").click(
                        lambda x=example: x, outputs=t5_input
                    )
    
    with gr.Tab("ℹ️ Info"):
        gr.Markdown("""
        ## About MyoMorph
        
        MyoMorph is a unified motion-language model that can:
        - Generate human motion from text descriptions
        - Generate text descriptions from motion data
        - Chat about motion and movement topics
        - Handle various motion-related tasks
        - Direct access to T5 language model backbone
        
        ### Usage Instructions:
        
        **Text-to-Motion:**
        1. Enter a description of human motion in natural language
        2. Click "Generate Motion" to create the motion
        3. View the generated motion video and download the motion data
        
        **Motion-to-Text:**
        1. Upload a motion file (.npy format) or use the sample motion
        2. Click "Generate Text" to get a description
        3. Read the generated natural language description
        
        **Chat:**
        1. Ask questions about human motion and movement
        2. Request motion descriptions or explanations
        3. Have conversations about various movement topics
        4. Get insights about human motion patterns
        
        **T5 Backbone:**
        1. Get detailed information about the underlying T5 model
        2. Use T5 directly for various text generation tasks
        3. Analyze how text gets tokenized (including motion tokens)
        4. Explore the 770M parameter language model directly
        
        ### File Formats:
        - Motion files should be in .npy format
        - Shape should be (frames, joints, 3) for 3D joint positions
        - Standard format has 22 joints representing human body
        
        ### Technical Details:
        - **Language Model:** T5-Base (770M parameters)
        - **Motion Tokenizer:** VQ-VAE with 512 motion tokens
        - **Vocabulary:** Extended T5 vocabulary + 515 motion tokens
        - **Architecture:** Encoder-Decoder (T5ForConditionalGeneration)
        
        ### Tips:
        - Use descriptive text for better motion generation
        - Motion descriptions work best with common human activities
        - Try asking specific questions in chat for detailed responses
        - Check the cache folder for saved motion files
        - The chat feature leverages motion-language knowledge for conversations
        - Use T5 Backbone tab to explore the underlying language model
        - Token analysis helps understand motion-text integration
        """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=8890,
        debug=True,
        share=True
    ) 