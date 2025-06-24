#!/usr/bin/env python3
"""
CLI Agent that uses fine-tuned model to generate and execute command-line plans.
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


class CLIAgent:
    def __init__(self, base_model_path, adapter_path, logs_dir="logs"):
        """
        Initialize the CLI agent with base model and LoRA adapter.
        
        Args:
            base_model_path: Path to base model (e.g., 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
            adapter_path: Path to fine-tuned LoRA adapter
            logs_dir: Directory to save execution logs
        """
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # --- Optimizations ---
        bnb_config = None
        torch_dtype = torch.float32
        attn_implementation = "eager"

        if self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            torch_dtype = torch.bfloat16
            
            if torch.cuda.get_device_capability()[0] >= 8:
                print("Flash Attention 2 is available. Using it for faster inference.")
                attn_implementation = "flash_attention_2"
            else:
                print("Flash Attention 2 not available, using standard attention.")
                attn_implementation = "sdpa"

        print("Loading base model with optimizations...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=True
        )
        
        print("Loading fine-tuned adapter...")
        self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
        
        print("Compiling model for faster execution...")
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True)
            print("Model compiled successfully.")
        except Exception as e:
            print(f"torch.compile failed: {e}. Running without compilation.")

        self.model.eval()
        
        # Initialize trace log
        self.trace_file = self.logs_dir / "trace.jsonl"
        
    def format_prompt(self, instruction):
        """Format the instruction as a chat prompt."""
        # Adjust this based on your model's chat format
        prompt = f"""<|user|>
{instruction}

Please provide a step-by-step plan to accomplish this task. If shell commands are needed, start each command line with the actual command.
<|assistant|>
"""
        return prompt
    
    def generate_plan(self, instruction, max_length=512, temperature=0.7):
        """
        Generate a step-by-step plan using the fine-tuned model.
        
        Args:
            instruction: Natural language instruction
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated plan as string
        """
        prompt = self.format_prompt(instruction)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new tokens (exclude the prompt)
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def extract_commands(self, plan_text):
        """
        Extract shell commands from the generated plan.
        Look for lines that start with common command patterns.
        """
        commands = []
        lines = plan_text.split('\n')
        
        # Common command patterns
        command_patterns = [
            r'^(git|bash|sh|ls|cd|mkdir|rm|cp|mv|tar|gzip|gunzip|grep|find|chmod|chown)\s+',
            r'^(python|pip|conda|virtualenv|source)\s+',
            r'^(sudo|su)\s+',
            r'^[a-zA-Z_][a-zA-Z0-9_]*\s+',  # Generic command pattern
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Remove step numbers like "1.", "Step 1:", etc.
            clean_line = re.sub(r'^\d+[\.\)]\s*', '', line)
            clean_line = re.sub(r'^Step\s+\d+:\s*', '', clean_line, flags=re.IGNORECASE)
            
            # Check if line matches command patterns
            for pattern in command_patterns:
                if re.match(pattern, clean_line):
                    commands.append(clean_line)
                    break
        
        return commands
    
    def execute_dry_run(self, command):
        """
        Execute command in dry-run mode (echo only).
        
        Args:
            command: Shell command to echo
            
        Returns:
            dict: Execution result with status and output
        """
        try:
            print(f"[DRY RUN] Would execute: {command}")
            
            # For demonstration, we'll actually run safe commands
            # In a real scenario, you might want to be more restrictive
            safe_commands = ['ls', 'pwd', 'whoami', 'date', 'echo']
            cmd_parts = command.split()
            
            if cmd_parts and cmd_parts[0] in safe_commands:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                return {
                    'command': command,
                    'status': 'executed',
                    'return_code': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            else:
                return {
                    'command': command,
                    'status': 'dry_run_only',
                    'return_code': 0,
                    'stdout': f"[DRY RUN] {command}",
                    'stderr': ''
                }
                
        except subprocess.TimeoutExpired:
            return {
                'command': command,
                'status': 'timeout',
                'return_code': -1,
                'stdout': '',
                'stderr': 'Command timed out'
            }
        except Exception as e:
            return {
                'command': command,
                'status': 'error',
                'return_code': -1,
                'stdout': '',
                'stderr': str(e)
            }
    
    def log_step(self, step_data):
        """Log a step to trace.jsonl file."""
        with open(self.trace_file, 'a', encoding='utf-8') as f:
            json.dump(step_data, f, ensure_ascii=False)
            f.write('\n')
    
    def process_instruction(self, instruction):
        """
        Main method to process a natural language instruction.
        
        Args:
            instruction: Natural language instruction
            
        Returns:
            dict: Processing results
        """
        timestamp = datetime.now().isoformat()
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n=== Processing Instruction ===")
        print(f"Instruction: {instruction}")
        print(f"Session ID: {session_id}")
        
        # Log initial step
        self.log_step({
            'session_id': session_id,
            'timestamp': timestamp,
            'step': 'input',
            'instruction': instruction
        })
        
        # Generate plan
        print("\n--- Generating Plan ---")
        plan = self.generate_plan(instruction)
        print(f"Generated Plan:\n{plan}")
        
        # Log plan generation
        self.log_step({
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'step': 'plan_generation',
            'plan': plan
        })
        
        # Extract commands
        commands = self.extract_commands(plan)
        print(f"\n--- Extracted Commands ---")
        for i, cmd in enumerate(commands, 1):
            print(f"{i}. {cmd}")
        
        # Execute commands in dry-run mode
        execution_results = []
        if commands:
            print(f"\n--- Executing Commands (Dry Run) ---")
            for i, command in enumerate(commands, 1):
                print(f"\nStep {i}: {command}")
                result = self.execute_dry_run(command)
                execution_results.append(result)
                
                # Log execution step
                self.log_step({
                    'session_id': session_id,
                    'timestamp': datetime.now().isoformat(),
                    'step': f'command_execution_{i}',
                    'command': command,
                    'result': result
                })
                
                if result['stdout']:
                    print(f"Output: {result['stdout']}")
                if result['stderr']:
                    print(f"Error: {result['stderr']}")
        
        # Final summary
        summary = {
            'session_id': session_id,
            'instruction': instruction,
            'plan': plan,
            'commands_extracted': len(commands),
            'commands': commands,
            'execution_results': execution_results
        }
        
        # Log summary
        self.log_step({
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'step': 'summary',
            'summary': summary
        })
        
        return summary


def log_to_dynamic_eval(prompt, command, score, notes):
    """Log the command execution to eval_dynamic.md"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    eval_file = Path("eval_dynamic.md")
    
    # Read existing content if file exists
    content = []
    if eval_file.exists():
        with open(eval_file, 'r') as f:
            content = f.readlines()
    
    # Extract existing table rows
    table_start = 0
    table_end = 0
    for i, line in enumerate(content):
        if '| # | Prompt |' in line:
            table_start = i + 2  # Skip header and separator lines
        if i > table_start and not line.strip() and not table_end:
            table_end = i
    
    # Add new row
    new_row = f"| {table_start - table_start + 1} | {prompt} | `{command}` | {score} | {notes} | {timestamp} |\n"
    
    # Insert new row after table start
    if table_start > 0 and table_end > 0:
        content.insert(table_end, new_row)
    else:
        # If no table exists, create one
        content = [
            "# Dynamic Agent Evaluation\n\n",
            "*Generated on: " + timestamp + "*\n\n",
            "## Overview\n",
            "This document tracks the agent's performance on various tasks with a scoring system from 0-2:\n",
            "- 0: Incorrect or non-functional command\n",
            "- 1: Partially correct but has issues\n",
            "- 2: Correct and functional command\n\n",
            "## Evaluation Results\n\n",
            "| # | Prompt | Generated Command | Score | Notes | Timestamp |\n",
            "|---|--------|-------------------|-------|-------|-----------|\n",
            new_row,
            "\n## Scoring Criteria\n",
            "- **2 points**: Command is correct, safe, and achieves the intended task\n",
            "- **1 point**: Command is partially correct but may have minor issues or be incomplete\n",
            "- **0 points**: Command is incorrect, unsafe, or doesn't achieve the intended task\n"
        ]
    
    # Write back to file
    with open(eval_file, 'w') as f:
        f.writelines(content)

def main():
    parser = argparse.ArgumentParser(description='CLI Agent for command generation')
    parser.add_argument('prompt', type=str, nargs='?', help='Natural language description of the command to generate')
    parser.add_argument('--model-path', type=str, default='./lora_adapter', help='Path to the fine-tuned model adapter')
    parser.add_argument('--base-model', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                       help='Base model to use')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without executing them')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    args = parser.parse_args()
    
    if not any(vars(args).values()) and len(sys.argv) == 1:
        parser.print_help()
        return
    
    try:
        # Initialize agent
        agent = CLIAgent(
            base_model_path=args.base_model,
            adapter_path=args.model_path,
            logs_dir='logs'
        )
        
        if args.interactive:
            print("=== Interactive CLI Agent ===")
            print("Enter natural language instructions (type 'quit' to exit):")
            
            while True:
                try:
                    instruction = input("\n> ").strip()
                    if instruction.lower() in ['quit', 'exit', 'q']:
                        break
                    if not instruction:
                        continue
                        
                    agent.process_instruction(instruction)
                    
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
                except Exception as e:
                    print(f"Error processing instruction: {e}")
                    
        elif args.instruction:
            result = agent.process_instruction(args.instruction)
            print(f"\n=== Summary ===")
            print(f"Commands extracted: {result['commands_extracted']}")
            print(f"Session logged to: {agent.trace_file}")
            
        else:
            print("Please provide --instruction or use --interactive mode")
            
    except Exception as e:
        print(f"Error initializing agent: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()