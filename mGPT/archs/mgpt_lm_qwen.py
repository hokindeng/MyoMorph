"""
MotionGPT Language Model with Qwen3-0.6B Backbone
Replaces T5-Base with more efficient and capable Qwen3-0.6B model
"""

import os
import math
import time
import random
import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import List, Optional, Union

from .tools.token_emb import NewTokenEmb


class QwenMLM(nn.Module):
    """
    Motion Language Model using Qwen3-0.6B as backbone
    Integrates motion tokens into Qwen3's vocabulary for motion-language understanding
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-0.6B",
        stage: str = "lm_pretrain",
        new_token_type: str = "insert",
        motion_codebook_size: int = 512,
        framerate: float = 20.0,
        down_t: int = 4,
        predict_ratio: float = 0.2,
        inbetween_ratio: float = 0.25,
        max_length: int = 256,
        quota_ratio: float = 0.5,
        enable_thinking: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        # Parameters
        self.m_codebook_size = motion_codebook_size
        self.max_length = max_length
        self.framerate = framerate
        self.down_t = down_t
        self.predict_ratio = predict_ratio
        self.inbetween_ratio = inbetween_ratio
        self.quota_ratio = quota_ratio
        self.stage = stage
        self.enable_thinking = enable_thinking

        # Load Qwen3 model and tokenizer
        print(f"🔥 Loading Qwen3-0.6B from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.language_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Set model type for decoder-only architecture
        self.lm_type = 'dec'
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Add motion tokens to vocabulary
        motion_tokens = [f'<motion_id_{i}>' for i in range(self.m_codebook_size + 3)]
        self.tokenizer.add_tokens(motion_tokens)
        
        print(f"📈 Extended vocabulary: {len(self.tokenizer)} tokens")
        print(f"   - Original Qwen3: {len(self.tokenizer) - len(motion_tokens)} tokens")
        print(f"   - Motion tokens: {self.m_codebook_size} (0-{self.m_codebook_size-1})")
        print(f"   - Special tokens: 3 (start, end, mask)")

        # Resize model embeddings for new tokens
        if new_token_type == "insert":
            self.language_model.resize_token_embeddings(len(self.tokenizer))
        elif new_token_type == "mlp":
            # Use specialized embedding for motion tokens
            shared = NewTokenEmb(
                self.language_model.get_input_embeddings(),
                self.m_codebook_size + 3
            )
            self.language_model.resize_token_embeddings(len(self.tokenizer))
            self.language_model.set_input_embeddings(shared)

        print(f"✅ Qwen3 backbone loaded successfully!")

    def forward(self, texts: List[str], motion_tokens: Tensor, lengths: List[int], tasks: dict):
        """Forward pass for training with motion-text pairs"""
        return self.forward_dec(texts, motion_tokens, lengths, tasks)

    def forward_dec(self, texts: List[str], motion_tokens: Tensor, lengths: List[int], tasks: dict):
        """
        Decoder-only forward pass for Qwen3
        Handles motion-language training with causal language modeling
        """
        # Convert motion tokens to strings
        motion_strings = self.motion_token_to_string(motion_tokens, lengths)

        # Training condition selection
        condition = random.choice(['supervised', 'supervised', 'supervised'])

        if condition == 'text':
            labels = texts
        elif condition == 'motion':
            labels = motion_strings
        else:
            # Use template system for structured training
            inputs, outputs = self.template_fulfill(tasks, lengths, motion_strings, texts)
            labels = []
            for i in range(len(inputs)):
                # Format as: <|user|>INPUT<|assistant|>OUTPUT<|end|>
                formatted_text = self._format_chat_template(inputs[i], outputs[i])
                labels.append(formatted_text)

        # Tokenize inputs
        encoding = self.tokenizer(
            labels,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        input_ids = encoding.input_ids.to(motion_tokens.device)
        attention_mask = encoding.attention_mask.to(motion_tokens.device)

        # Causal language modeling loss
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids  # For CLM, labels are the same as input_ids
        )

        return outputs

    def _format_chat_template(self, input_text: str, output_text: str) -> str:
        """Format input-output pair using Qwen3 chat template"""
        messages = [
            {"role": "user", "content": input_text}
        ]
        
        # Apply chat template
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking
        )
        
        # Add the expected output
        formatted = formatted + output_text + self.tokenizer.eos_token
        
        return formatted

    def generate_conditional(
        self,
        texts: Optional[List[str]] = None,
        motion_tokens: Optional[Tensor] = None,
        lengths: Optional[List[int]] = None,
        task: str = "t2m",
        with_len: bool = False,
        stage: str = 'test',
        tasks: dict = None
    ):
        """Generate motion or text using Qwen3 backbone"""
        
        self.device = self.language_model.device

        if task in ["t2m", "m2m", "pred", "inbetween"]:
            # Motion generation tasks
            if task == "t2m":
                assert texts is not None
                motion_strings = [''] * len(texts)
                if not with_len:
                    if tasks is None:
                        tasks = [{
                            'input': ['Generate motion: <Caption_Placeholder>'],
                            'output': ['']
                        }] * len(texts)
                    lengths = [0] * len(texts)
                else:
                    tasks = [{
                        'input': [
                            'Generate motion with <Frame_Placeholder> frames: <Caption_Placeholder>'
                        ],
                        'output': ['']
                    }] * len(texts)

            elif task == "pred":
                assert motion_tokens is not None and lengths is not None
                texts = [''] * len(lengths)
                tasks = [{
                    'input': ['Predict motion: <Motion_Placeholder_s1>'],
                    'output': ['']
                }] * len(lengths)

                motion_strings_old = self.motion_token_to_string(motion_tokens, lengths)
                motion_strings = []
                for i, length in enumerate(lengths):
                    split = length // 5
                    motion_strings.append(
                        '>'.join(motion_strings_old[i].split('>')[:split]) + '>'
                    )

            elif task == "inbetween":
                assert motion_tokens is not None and lengths is not None
                texts = [''] * len(lengths)
                tasks = [{
                    'input': [
                        "Complete the masked motion: <Motion_Placeholder_Masked>"
                    ],
                    'output': ['']
                }] * len(lengths)
                motion_strings = self.motion_token_to_string(motion_tokens, lengths)

            # Generate input prompts
            inputs, outputs = self.template_fulfill(tasks, lengths, motion_strings, texts, stage)

            # Format as chat messages
            formatted_inputs = []
            for input_text in inputs:
                messages = [{"role": "user", "content": input_text}]
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking
                )
                formatted_inputs.append(formatted)

            # Generate responses
            outputs_tokens, cleaned_text = self.generate_direct(
                formatted_inputs,
                max_length=512,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                top_k=20
            )

            return outputs_tokens

        elif task == "m2t":
            # Motion to text generation
            assert motion_tokens is not None and lengths is not None

            motion_strings = self.motion_token_to_string(motion_tokens, lengths)

            if not with_len:
                tasks = [{
                    'input': ['Generate text: <Motion_Placeholder>'],
                    'output': ['']
                }] * len(lengths)
            else:
                tasks = [{
                    'input': [
                        'Generate text with <Frame_Placeholder> frames: <Motion_Placeholder>'
                    ],
                    'output': ['']
                }] * len(lengths)

            texts = [''] * len(lengths)
            inputs, outputs = self.template_fulfill(tasks, lengths, motion_strings, texts)

            # Format as chat messages
            formatted_inputs = []
            for input_text in inputs:
                messages = [{"role": "user", "content": input_text}]
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False  # Disable thinking for text generation
                )
                formatted_inputs.append(formatted)

            outputs_tokens, cleaned_text = self.generate_direct(
                formatted_inputs,
                max_length=128,
                do_sample=False,
                temperature=0.7,
                top_p=0.8
            )
            
            return cleaned_text

    def generate_direct(
        self,
        texts: List[str],
        max_length: int = 256,
        do_sample: bool = True,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        **kwargs
    ):
        """Direct text generation using Qwen3"""
        
        # Tokenize inputs
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True
        )

        input_ids = encoding.input_ids.to(self.device)
        attention_mask = encoding.attention_mask.to(self.device)

        # Generate with Qwen3
        with torch.no_grad():
            outputs = self.language_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_length,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        # Decode outputs
        generated_texts = []
        for i, output in enumerate(outputs):
            # Remove input tokens from output
            new_tokens = output[len(input_ids[i]):]
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            generated_texts.append(generated_text.strip())

        # Extract motion tokens from generated text
        outputs_tokens, cleaned_text = self.motion_string_to_token(generated_texts)

        return outputs_tokens, cleaned_text

    def motion_token_to_string(self, motion_token: Tensor, lengths: List[int]) -> List[str]:
        """Convert motion tensor to string representation"""
        motion_string = []
        for i in range(len(motion_token)):
            motion_i = motion_token[i].cpu() if motion_token[i].device.type == 'cuda' else motion_token[i]
            motion_list = motion_i.tolist()[:lengths[i]]
            motion_string.append(
                (f'<motion_id_{self.m_codebook_size}>' +
                 ''.join([f'<motion_id_{int(token)}>' for token in motion_list]) +
                 f'<motion_id_{self.m_codebook_size + 1}>')
            )
        return motion_string

    def motion_string_to_token(self, motion_string: List[str]):
        """Convert string representation back to motion tokens"""
        motion_tokens = []
        output_string = []
        
        for i in range(len(motion_string)):
            string = self.get_middle_str(
                motion_string[i], 
                f'<motion_id_{self.m_codebook_size}>',
                f'<motion_id_{self.m_codebook_size + 1}>'
            )
            string_list = string.split('><')
            token_list = [
                int(token.split('_')[-1].replace('>', ''))
                for token in string_list[1:-1]
                if token.startswith('motion_id_')
            ]
            
            if len(token_list) == 0:
                token_list = [0]
                
            token_list_padded = torch.tensor(token_list, dtype=int).to(self.device)
            motion_tokens.append(token_list_padded)
            output_string.append(motion_string[i].replace(string, '<Motion_Placeholder>'))

        return motion_tokens, output_string

    def placeholder_fulfill(self, prompt: str, length: int, motion_string: str, text: str):
        """Fill template placeholders with actual content"""
        seconds = math.floor(length / self.framerate)
        motion_splited = motion_string.split('>')
        token_length = length / self.down_t
        predict_head = int(token_length * self.predict_ratio + 1)
        masked_head = int(token_length * self.inbetween_ratio + 1)
        masked_tail = int(token_length * (1 - self.inbetween_ratio) + 1)
        
        motion_predict_head = '>'.join(
            motion_splited[:predict_head]
        ) + f'><motion_id_{self.m_codebook_size+1}>'
        
        motion_predict_last = f'<motion_id_{self.m_codebook_size}>' + '>'.join(
            motion_splited[predict_head:]
        )

        motion_masked = '>'.join(
            motion_splited[:masked_head]
        ) + '>' + f'<motion_id_{self.m_codebook_size+2}>' * (
            masked_tail - masked_head
        ) + '>'.join(motion_splited[masked_tail:])

        if random.random() < self.quota_ratio:
            text = f'"{text}"'

        prompt = prompt.replace('<Caption_Placeholder>', text).replace(
            '<Motion_Placeholder>', motion_string).replace(
            '<Frame_Placeholder>', f'{length}').replace(
            '<Second_Placeholder>', '%.1f' % seconds).replace(
            '<Motion_Placeholder_s1>', motion_predict_head).replace(
            '<Motion_Placeholder_s2>', motion_predict_last).replace(
            '<Motion_Placeholder_Masked>', motion_masked)

        return prompt

    def template_fulfill(self, tasks, lengths, motion_strings, texts, stage='test'):
        """Fulfill templates with actual content"""
        inputs = []
        outputs = []
        for i in range(len(lengths)):
            input_template = random.choice(tasks[i]['input'])
            output_template = random.choice(tasks[i]['output'])
            length = lengths[i]
            inputs.append(
                self.placeholder_fulfill(input_template, length, motion_strings[i], texts[i])
            )
            outputs.append(
                self.placeholder_fulfill(output_template, length, motion_strings[i], texts[i])
            )

        return inputs, outputs

    def get_middle_str(self, content, startStr, endStr):
        """Extract content between start and end strings"""
        try:
            startIndex = content.index(startStr)
            if startIndex >= 0:
                startIndex += len(startStr)
            endIndex = content.index(endStr)
        except:
            return f'<motion_id_{self.m_codebook_size}><motion_id_0><motion_id_{self.m_codebook_size+1}>'

        return f'<motion_id_{self.m_codebook_size}>' + content[startIndex:endIndex] + f'<motion_id_{self.m_codebook_size+1}>' 