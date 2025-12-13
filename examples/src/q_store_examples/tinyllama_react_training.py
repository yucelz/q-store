"""
TinyLlama React Fine-Tuning with Q-Store Quantum Database
Demonstrates quantum-enhanced ML training with intelligent data selection

Usage:
    Step 1: Generate dataset
        python react_dataset_generator.py
    
    Step 2: Verify dataset
        cat react_train.jsonl | wc -l  # Should show 3000+
    
    Step 3: Run quantum training
        python tinyllama_react_training.py
"""

import asyncio
import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from getpass import getpass
from dotenv import load_dotenv

# Q-Store imports
from q_store import QuantumDatabase, DatabaseConfig, QueryMode

# ML/Transformers imports (optional - only needed for actual training)
try:
    from datasets import Dataset
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        TrainingArguments, 
        Trainer
    )
    from peft import LoraConfig, get_peft_model
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers not installed. Install with: pip install transformers peft datasets torch")

# Load environment variables
load_dotenv()


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Model settings
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_dir: str = "./tinyllama-react-quantum"
    
    # Data settings
    training_data_file: str = "react_train.jsonl"
    max_samples: int = 1000
    embedding_dim: int = 768
    
    # Training hyperparameters
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    max_length: int = 1024
    
    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Quantum-enhanced features
    use_quantum_sampling: bool = True
    use_curriculum_learning: bool = True
    use_hard_negative_mining: bool = True
    coherence_time: float = 5000.0  # ms


# ============================================================================
# Embedding Generation (Mock for Demo)
# ============================================================================

def generate_text_embedding(text: str, dim: int = 768) -> np.ndarray:
    """
    Generate mock embedding for text.
    In production, use sentence-transformers or OpenAI embeddings.
    """
    # Simple hash-based embedding for demo
    np.random.seed(abs(hash(text)) % (2**32))
    embedding = np.random.randn(dim)
    
    # Add some structure based on text length and content
    embedding[0] = len(text) / 1000.0  # Length signal
    embedding[1] = text.count('function') * 0.1  # Function signal
    embedding[2] = text.count('class') * 0.1  # Class signal
    
    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding


def classify_instruction_type(instruction: str) -> str:
    """Classify instruction type for context assignment"""
    instruction_lower = instruction.lower()
    
    if any(word in instruction_lower for word in ['create', 'build', 'make', 'generate']):
        return 'generation'
    elif any(word in instruction_lower for word in ['fix', 'debug', 'error', 'issue']):
        return 'debugging'
    elif any(word in instruction_lower for word in ['explain', 'what', 'how', 'why']):
        return 'explanation'
    elif any(word in instruction_lower for word in ['convert', 'transform', 'change']):
        return 'conversion'
    elif any(word in instruction_lower for word in ['optimize', 'improve', 'enhance']):
        return 'optimization'
    else:
        return 'general'


def estimate_difficulty(sample: Dict[str, Any]) -> str:
    """Estimate sample difficulty for curriculum learning"""
    output_len = len(sample.get('output', ''))
    instruction_len = len(sample.get('instruction', ''))
    
    # Simple heuristic: longer outputs = harder
    if output_len > 500 or instruction_len > 200:
        return 'hard'
    elif output_len > 200 or instruction_len > 100:
        return 'medium'
    else:
        return 'easy'


# ============================================================================
# Quantum Database Integration
# ============================================================================

class QuantumTrainingDataManager:
    """Manages training data using Q-Store quantum database"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.db: Optional[QuantumDatabase] = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize quantum database"""
        print("🔮 Initializing Q-Store quantum database...")
        
        # Get API keys
        pinecone_key = os.getenv('PINECONE_API_KEY')
        ionq_key = os.getenv('IONQ_API_KEY')
        pinecone_environment = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
        
        if not pinecone_key:
            raise ValueError(
                "PINECONE_API_KEY is required. Please add it to your .env file.\n"
                "Get your API key from: https://www.pinecone.io/"
            )
        
        # Create database configuration
        db_config = DatabaseConfig(
            # Pinecone settings
            pinecone_api_key=pinecone_key,
            pinecone_environment=pinecone_environment,
            pinecone_index_name="tinyllama-react-training",
            pinecone_dimension=self.config.embedding_dim,
            pinecone_metric="cosine",
            
            # IonQ quantum settings (optional)
            ionq_api_key=ionq_key,
            ionq_target="simulator",
            
            # Feature flags
            enable_quantum=True if ionq_key else False,
            enable_superposition=True if ionq_key else False,
            enable_entanglement=True,
            enable_tunneling=True if ionq_key else False,
            
            # Performance tuning
            default_coherence_time=self.config.coherence_time,
            max_quantum_states=1000,
            classical_candidate_pool=500
        )
        
        # Initialize database
        self.db = QuantumDatabase(config=db_config)
        await self.db.initialize()
        
        self.initialized = True
        print("✓ Quantum database initialized\n")
    
    async def load_training_data(self, jsonl_file: str):
        """Load training data from JSONL file into quantum database"""
        if not self.initialized:
            await self.initialize()
        
        print(f"📚 Loading training data from {jsonl_file}...")
        
        if not Path(jsonl_file).exists():
            print(f"⚠️  File not found: {jsonl_file}")
            print("   Attempting to generate dataset using react_dataset_generator.py...")
            self._generate_dataset(jsonl_file)
        
        samples = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        
        # Limit samples if specified
        if self.config.max_samples:
            samples = samples[:self.config.max_samples]
        
        print(f"  Found {len(samples)} training samples")
        print("  Storing in quantum database with superposition contexts...")
        
        # Store each sample with quantum features
        for idx, sample in enumerate(samples):
            # Generate embedding
            text = sample.get('instruction', '') + ' ' + sample.get('output', '')
            embedding = generate_text_embedding(text, self.config.embedding_dim)
            
            # Classify instruction type
            inst_type = classify_instruction_type(sample.get('instruction', ''))
            
            # Estimate difficulty
            difficulty = estimate_difficulty(sample)
            
            # Store with multiple contexts (superposition)
            contexts = [
                (inst_type, 0.6),  # Primary context
                ('general', 0.3),  # General context
                (difficulty, 0.1)  # Difficulty context
            ]
            
            await self.db.insert(
                id=f'sample_{idx}',
                vector=embedding,
                contexts=contexts,
                coherence_time=self.config.coherence_time,
                metadata={
                    'instruction': sample.get('instruction', ''),
                    'input': sample.get('input', ''),
                    'output': sample.get('output', ''),
                    'instruction_type': inst_type,
                    'difficulty': difficulty,
                    'length': len(sample.get('output', ''))
                }
            )
            
            if (idx + 1) % 100 == 0:
                print(f"    Stored {idx + 1}/{len(samples)} samples")
        
        # Create entangled groups by instruction type
        print("\n🔗 Creating entangled groups by instruction type...")
        
        instruction_types = set(classify_instruction_type(s.get('instruction', '')) for s in samples)
        for inst_type in instruction_types:
            group_ids = [
                f'sample_{idx}' for idx, s in enumerate(samples)
                if classify_instruction_type(s.get('instruction', '')) == inst_type
            ]
            
            if len(group_ids) > 1:
                self.db.create_entangled_group(
                    group_id=f'group_{inst_type}',
                    entity_ids=group_ids,
                    correlation_strength=0.85
                )
                print(f"  ✓ Entangled {len(group_ids)} samples in '{inst_type}' group")
        
        print(f"\n✅ Loaded {len(samples)} samples into quantum database\n")
        return len(samples)
    
    async def sample_training_batch(
        self,
        batch_size: int,
        epoch: int = 0,
        model_state: Optional[np.ndarray] = None,
        context: str = 'general',
        use_curriculum: bool = True,
        use_tunneling: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Sample training batch using quantum-enhanced selection
        
        Args:
            batch_size: Number of samples to retrieve
            epoch: Current training epoch (for curriculum learning)
            model_state: Current model embedding state (for relevance)
            context: Instruction type context to focus on
            use_curriculum: Apply curriculum learning (easy -> hard)
            use_tunneling: Use quantum tunneling for exploration
        """
        if not self.initialized:
            await self.initialize()
        
        # Use model state or random query vector
        if model_state is None:
            model_state = np.random.randn(self.config.embedding_dim)
            model_state = model_state / np.linalg.norm(model_state)
        
        # Adjust query mode based on epoch and settings
        if use_curriculum:
            # Early epochs: easy samples (PRECISE)
            # Later epochs: harder samples (EXPLORATORY)
            if epoch < 1:
                mode = QueryMode.PRECISE
                difficulty_context = 'easy'
            elif epoch < 2:
                mode = QueryMode.BALANCED
                difficulty_context = 'medium'
            else:
                mode = QueryMode.EXPLORATORY
                difficulty_context = 'hard'
        else:
            mode = QueryMode.BALANCED
            difficulty_context = context
        
        # Query quantum database
        results = await self.db.query(
            vector=model_state,
            context=difficulty_context if use_curriculum else context,
            enable_tunneling=use_tunneling,
            mode=mode,
            top_k=batch_size
        )
        
        # Extract samples
        batch = []
        for result in results:
            batch.append({
                'instruction': result.metadata.get('instruction', ''),
                'input': result.metadata.get('input', ''),
                'output': result.metadata.get('output', ''),
                'quantum_score': result.score,
                'difficulty': result.metadata.get('difficulty', 'unknown')
            })
        
        return batch
    
    async def hard_negative_mining(
        self,
        model_state: np.ndarray,
        context: str,
        num_negatives: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find hard negative examples using quantum tunneling
        """
        if not self.initialized:
            await self.initialize()
        
        # Use tunneling to find challenging examples
        results = await self.db.query(
            vector=model_state,
            context=context,
            enable_tunneling=True,
            mode=QueryMode.EXPLORATORY,
            top_k=num_negatives
        )
        
        negatives = []
        for result in results:
            negatives.append({
                'instruction': result.metadata.get('instruction', ''),
                'input': result.metadata.get('input', ''),
                'output': result.metadata.get('output', ''),
                'difficulty': result.metadata.get('difficulty', 'unknown')
            })
        
        return negatives
    
    async def close(self):
        """Close database connection"""
        if self.db:
            await self.db.close()
    
    def _generate_dataset(self, output_file: str):
        """Generate React training dataset using react_dataset_generator.py"""
        generator_path = Path(__file__).parent.parent / 'react_dataset_generator.py'
        
        # Check if generator exists in parent directory or Downloads
        if not generator_path.exists():
            # Try Downloads folder
            generator_path = Path.home() / 'Downloads' / 'react_dataset_generator.py'
        
        if not generator_path.exists():
            print(f"   ⚠️  react_dataset_generator.py not found!")
            print(f"   Please ensure react_dataset_generator.py is in:")
            print(f"     - {Path(__file__).parent.parent}")
            print(f"     - {Path.home() / 'Downloads'}")
            print(f"   Creating minimal sample data instead...")
            self._create_sample_data(output_file)
            return
        
        print(f"   ✓ Found generator: {generator_path}")
        print(f"   🔨 Generating dataset...")
        
        # Import and run the generator
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("react_dataset_generator", generator_path)
            generator_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(generator_module)
            
            # Create generator instance and generate dataset
            generator = generator_module.ReactDatasetGenerator()
            generator.generate_all()
            generator.save_to_jsonl(str(output_file))
            
            print(f"   ✅ Dataset generated successfully!")
            
        except Exception as e:
            print(f"   ⚠️  Error generating dataset: {e}")
            print(f"   Creating minimal sample data instead...")
            self._create_sample_data(output_file)
    
    def _create_sample_data(self, output_file: str):
        """Create sample React training data for demo"""
        samples = [
            {
                "instruction": "Create a React counter component using hooks",
                "input": "",
                "output": "import React, { useState } from 'react';\n\nfunction Counter() {\n  const [count, setCount] = useState(0);\n\n  return (\n    <div>\n      <h1>Count: {count}</h1>\n      <button onClick={() => setCount(count + 1)}>Increment</button>\n      <button onClick={() => setCount(count - 1)}>Decrement</button>\n    </div>\n  );\n}\n\nexport default Counter;"
            },
            {
                "instruction": "Build a React login form with email and password",
                "input": "",
                "output": "import React, { useState } from 'react';\n\nfunction LoginForm() {\n  const [email, setEmail] = useState('');\n  const [password, setPassword] = useState('');\n\n  const handleSubmit = (e) => {\n    e.preventDefault();\n    console.log('Login:', { email, password });\n  };\n\n  return (\n    <form onSubmit={handleSubmit}>\n      <input\n        type=\"email\"\n        value={email}\n        onChange={(e) => setEmail(e.target.value)}\n        placeholder=\"Email\"\n        required\n      />\n      <input\n        type=\"password\"\n        value={password}\n        onChange={(e) => setPassword(e.target.value)}\n        placeholder=\"Password\"\n        required\n      />\n      <button type=\"submit\">Login</button>\n    </form>\n  );\n}\n\nexport default LoginForm;"
            },
            {
                "instruction": "Explain the useState hook in React",
                "input": "",
                "output": "useState is a Hook that lets you add React state to function components. It returns an array with two values: the current state and a function to update it. You call useState with the initial state value, and it preserves this value between re-renders. When you call the setter function, React re-renders the component with the new state value."
            },
            {
                "instruction": "Create a todo list component in React",
                "input": "",
                "output": "import React, { useState } from 'react';\n\nfunction TodoList() {\n  const [todos, setTodos] = useState([]);\n  const [input, setInput] = useState('');\n\n  const addTodo = () => {\n    if (input.trim()) {\n      setTodos([...todos, { id: Date.now(), text: input, completed: false }]);\n      setInput('');\n    }\n  };\n\n  const toggleTodo = (id) => {\n    setTodos(todos.map(todo =>\n      todo.id === id ? { ...todo, completed: !todo.completed } : todo\n    ));\n  };\n\n  return (\n    <div>\n      <input\n        value={input}\n        onChange={(e) => setInput(e.target.value)}\n        placeholder=\"Add todo\"\n      />\n      <button onClick={addTodo}>Add</button>\n      <ul>\n        {todos.map(todo => (\n          <li\n            key={todo.id}\n            onClick={() => toggleTodo(todo.id)}\n            style={{ textDecoration: todo.completed ? 'line-through' : 'none' }}\n          >\n            {todo.text}\n          </li>\n        ))}\n      </ul>\n    </div>\n  );\n}\n\nexport default TodoList;"
            },
            {
                "instruction": "Fix the infinite loop in this useEffect",
                "input": "useEffect(() => {\n  setCount(count + 1);\n});",
                "output": "The infinite loop occurs because useEffect runs after every render, and calling setCount causes a re-render. To fix it, add a dependency array:\n\nuseEffect(() => {\n  setCount(count + 1);\n}, []); // Empty array means it runs only once\n\nOr if you want it to run when count changes:\n\nuseEffect(() => {\n  // Your logic here\n}, [count]); // Only runs when count changes"
            }
        ]
        
        with open(output_file, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
        
        print(f"  ✓ Created sample data: {output_file}")


# ============================================================================
# Prompt Formatting
# ============================================================================

def format_prompt(example: Dict[str, Any]) -> str:
    """Format training sample for TinyLlama chat format"""
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output = example.get('output', '')
    
    if input_text:
        prompt = f"""<|user|>
{instruction}

{input_text}
<|assistant|>
{output}"""
    else:
        prompt = f"""<|user|>
{instruction}
<|assistant|>
{output}"""
    
    return prompt


# ============================================================================
# Training Pipeline (if transformers available)
# ============================================================================

async def train_with_quantum_database(config: TrainingConfig):
    """Main training pipeline using Q-Store quantum database"""
    
    print("=" * 70)
    print("TinyLlama React Fine-Tuning with Q-Store Quantum Database")
    print("=" * 70)
    print()
    
    # Initialize quantum data manager
    data_manager = QuantumTrainingDataManager(config)
    
    try:
        # Load training data into quantum database
        await data_manager.load_training_data(config.training_data_file)
        
        # Demo: Sample batches with different strategies
        print("🎯 Demonstrating quantum-enhanced data sampling:\n")
        
        # 1. Curriculum learning: easy samples
        print("1. Curriculum Learning (Epoch 0 - Easy samples):")
        easy_batch = await data_manager.sample_training_batch(
            batch_size=3,
            epoch=0,
            use_curriculum=True
        )
        for i, sample in enumerate(easy_batch, 1):
            print(f"   {i}. [{sample['difficulty']}] {sample['instruction'][:60]}...")
        print()
        
        # 2. Curriculum learning: medium samples
        print("2. Curriculum Learning (Epoch 1 - Medium samples):")
        medium_batch = await data_manager.sample_training_batch(
            batch_size=3,
            epoch=1,
            use_curriculum=True
        )
        for i, sample in enumerate(medium_batch, 1):
            print(f"   {i}. [{sample['difficulty']}] {sample['instruction'][:60]}...")
        print()
        
        # 3. Context-specific sampling
        print("3. Context-Specific Sampling (Generation tasks):")
        gen_batch = await data_manager.sample_training_batch(
            batch_size=3,
            context='generation',
            use_curriculum=False
        )
        for i, sample in enumerate(gen_batch, 1):
            print(f"   {i}. {sample['instruction'][:60]}...")
        print()
        
        # 4. Hard negative mining with tunneling
        print("4. Hard Negative Mining (with quantum tunneling):")
        model_state = np.random.randn(config.embedding_dim)
        model_state = model_state / np.linalg.norm(model_state)
        
        hard_negatives = await data_manager.hard_negative_mining(
            model_state=model_state,
            context='debugging',
            num_negatives=3
        )
        for i, sample in enumerate(hard_negatives, 1):
            print(f"   {i}. {sample['instruction'][:60]}...")
        print()
        
        # Check if transformers is available for actual training
        if not TRANSFORMERS_AVAILABLE:
            print("⚠️  Transformers not available. Skipping actual model training.")
            print("   Install with: pip install transformers peft datasets torch")
            print("\n✅ Quantum database integration demo completed successfully!")
            return
        
        # Continue with actual training if transformers available
        print("🤖 Loading TinyLlama model...")
        
        tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        print("✓ Model loaded")
        
        # Apply LoRA
        print("\n🔧 Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Sample training data from quantum database
        print("\n📊 Sampling training data from quantum database...")
        all_samples = await data_manager.sample_training_batch(
            batch_size=config.max_samples,
            use_curriculum=False
        )
        
        # Format prompts
        formatted_samples = [
            {"text": format_prompt(sample)} for sample in all_samples
        ]
        
        # Create HuggingFace dataset
        dataset = Dataset.from_list(formatted_samples)
        
        # Tokenize
        print("🔤 Tokenizing dataset...")
        
        def tokenize(example):
            return tokenizer(
                example["text"],
                truncation=True,
                max_length=config.max_length,
                padding="max_length"
            )
        
        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Training configuration
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            num_train_epochs=config.num_train_epochs,
            learning_rate=config.learning_rate,
            fp16=True,
            logging_steps=50,
            save_strategy="epoch",
            save_total_limit=2,
            report_to="none",
            warmup_steps=100
        )
        
        # Train
        print("\n🚀 Starting training with quantum-enhanced dataset...")
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset
        )
        
        trainer.train()
        
        # Save model
        print("\n💾 Saving fine-tuned model...")
        model.save_pretrained(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
        
        print(f"\n✅ Training complete! Model saved to {config.output_dir}")
        
    finally:
        # Cleanup
        await data_manager.close()
        print("\n🎉 All done!")


# ============================================================================
# Main Execution
# ============================================================================

async def main():
    """Main entry point"""
    
    print("=" * 70)
    print("TinyLlama React Fine-Tuning with Q-Store Quantum Database")
    print("=" * 70)
    print()
    print("📋 WORKFLOW:")
    print("   1. Check/Generate React training dataset (react_train.jsonl)")
    print("   2. Load dataset into quantum database")
    print("   3. Demonstrate quantum-enhanced sampling strategies")
    print("   4. [Optional] Train TinyLlama model if transformers installed")
    print()
    print("=" * 70)
    print()
    
    # Configuration
    config = TrainingConfig(
        training_data_file="react_train.jsonl",
        output_dir="./tinyllama-react-quantum",
        max_samples=1000,  # Use more samples for better training
        use_quantum_sampling=True,
        use_curriculum_learning=True,
        use_hard_negative_mining=True
    )
    
    # Run training
    await train_with_quantum_database(config)


if __name__ == "__main__":
    asyncio.run(main())
