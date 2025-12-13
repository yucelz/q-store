"""
React Training Dataset Generator for Medium-Level Fine-Tuning
Creates high-quality React training samples in JSONL format
"""

import json
import random
from typing import List, Dict
from pathlib import Path

# ============================================================================
# DATASET TEMPLATES - MEDIUM LEVEL COMPLEXITY
# ============================================================================

class ReactDatasetGenerator:
    """Generates medium-level React training samples"""
    
    def __init__(self):
        self.samples = []
        
    # ------------------------------------------------------------------------
    # COMPONENT GENERATION SAMPLES (40%)
    # ------------------------------------------------------------------------
    
    def generate_component_samples(self, count: int = 1200):
        """Generate React component creation samples"""
        
        templates = [
            # useState Hook Components
            {
                "instruction": "Create a React counter component with increment and decrement buttons",
                "input": "",
                "output": """import { useState } from 'react';

export default function Counter() {
  const [count, setCount] = useState(0);

  return (
    <div className="counter">
      <h2>Count: {count}</h2>
      <button onClick={() => setCount(count + 1)}>Increment</button>
      <button onClick={() => setCount(count - 1)}>Decrement</button>
      <button onClick={() => setCount(0)}>Reset</button>
    </div>
  );
}"""
            },
            {
                "instruction": "Build a React todo list component with add and remove functionality",
                "input": "",
                "output": """import { useState } from 'react';

export default function TodoList() {
  const [todos, setTodos] = useState([]);
  const [input, setInput] = useState('');

  const addTodo = () => {
    if (input.trim()) {
      setTodos([...todos, { id: Date.now(), text: input }]);
      setInput('');
    }
  };

  const removeTodo = (id) => {
    setTodos(todos.filter(todo => todo.id !== id));
  };

  return (
    <div>
      <input 
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Enter todo"
      />
      <button onClick={addTodo}>Add</button>
      <ul>
        {todos.map(todo => (
          <li key={todo.id}>
            {todo.text}
            <button onClick={() => removeTodo(todo.id)}>Delete</button>
          </li>
        ))}
      </ul>
    </div>
  );
}"""
            },
            {
                "instruction": "Create a React form component with validation for email and password",
                "input": "",
                "output": """import { useState } from 'react';

export default function LoginForm() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [errors, setErrors] = useState({});

  const validate = () => {
    const newErrors = {};
    if (!email.includes('@')) {
      newErrors.email = 'Invalid email format';
    }
    if (password.length < 6) {
      newErrors.password = 'Password must be at least 6 characters';
    }
    return newErrors;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const validationErrors = validate();
    if (Object.keys(validationErrors).length === 0) {
      console.log('Form submitted', { email, password });
    } else {
      setErrors(validationErrors);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        placeholder="Email"
      />
      {errors.email && <span className="error">{errors.email}</span>}
      
      <input
        type="password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        placeholder="Password"
      />
      {errors.password && <span className="error">{errors.password}</span>}
      
      <button type="submit">Login</button>
    </form>
  );
}"""
            },
            {
                "instruction": "Build a React search filter component for a list of items",
                "input": "",
                "output": """import { useState } from 'react';

export default function SearchFilter({ items }) {
  const [searchTerm, setSearchTerm] = useState('');

  const filteredItems = items.filter(item =>
    item.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div>
      <input
        type="text"
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        placeholder="Search..."
      />
      <ul>
        {filteredItems.map((item, index) => (
          <li key={index}>{item}</li>
        ))}
      </ul>
      <p>Showing {filteredItems.length} of {items.length} items</p>
    </div>
  );
}"""
            },
            {
                "instruction": "Create a React accordion component that expands and collapses sections",
                "input": "",
                "output": """import { useState } from 'react';

export default function Accordion({ sections }) {
  const [activeIndex, setActiveIndex] = useState(null);

  const toggleSection = (index) => {
    setActiveIndex(activeIndex === index ? null : index);
  };

  return (
    <div className="accordion">
      {sections.map((section, index) => (
        <div key={index} className="accordion-item">
          <button
            className="accordion-header"
            onClick={() => toggleSection(index)}
          >
            {section.title}
          </button>
          {activeIndex === index && (
            <div className="accordion-content">
              {section.content}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}"""
            },
            {
                "instruction": "Build a React modal component with open and close functionality",
                "input": "",
                "output": """import { useState } from 'react';

export default function Modal({ trigger, children }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <div onClick={() => setIsOpen(true)}>
        {trigger}
      </div>
      
      {isOpen && (
        <div className="modal-overlay" onClick={() => setIsOpen(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button 
              className="modal-close"
              onClick={() => setIsOpen(false)}
            >
              ×
            </button>
            {children}
          </div>
        </div>
      )}
    </>
  );
}"""
            },
            {
                "instruction": "Create a React tabs component that switches between different content",
                "input": "",
                "output": """import { useState } from 'react';

export default function Tabs({ tabs }) {
  const [activeTab, setActiveTab] = useState(0);

  return (
    <div className="tabs">
      <div className="tab-headers">
        {tabs.map((tab, index) => (
          <button
            key={index}
            className={activeTab === index ? 'active' : ''}
            onClick={() => setActiveTab(index)}
          >
            {tab.title}
          </button>
        ))}
      </div>
      <div className="tab-content">
        {tabs[activeTab].content}
      </div>
    </div>
  );
}"""
            }
        ]
        
        # Duplicate and vary templates to reach target count
        for _ in range(count // len(templates)):
            self.samples.extend(templates)
        
    # ------------------------------------------------------------------------
    # BUG FIXING SAMPLES (25%)
    # ------------------------------------------------------------------------
    
    def generate_bug_fixing_samples(self, count: int = 750):
        """Generate bug fixing samples"""
        
        bug_samples = [
            {
                "instruction": "Fix this React component that has a missing key prop in the list",
                "input": """function List({ items }) {
  return (
    <ul>
      {items.map(item => <li>{item}</li>)}
    </ul>
  );
}""",
                "output": """function List({ items }) {
  return (
    <ul>
      {items.map((item, index) => <li key={index}>{item}</li>)}
    </ul>
  );
}

// Note: If items have unique IDs, use those instead of index
// Example: <li key={item.id}>{item.name}</li>"""
            },
            {
                "instruction": "Fix this component where state update doesn't trigger re-render",
                "input": """import { useState } from 'react';

function UserProfile() {
  const [user, setUser] = useState({ name: 'John', age: 30 });
  
  const updateAge = () => {
    user.age = 31;
    setUser(user);
  };
  
  return <div>{user.name} is {user.age}</div>;
}""",
                "output": """import { useState } from 'react';

function UserProfile() {
  const [user, setUser] = useState({ name: 'John', age: 30 });
  
  const updateAge = () => {
    // Create a new object to trigger re-render
    setUser({ ...user, age: 31 });
  };
  
  return <div>{user.name} is {user.age}</div>;
}"""
            },
            {
                "instruction": "Fix the infinite loop caused by useEffect without dependencies",
                "input": """import { useState, useEffect } from 'react';

function DataFetcher() {
  const [data, setData] = useState(null);
  
  useEffect(() => {
    fetch('/api/data')
      .then(res => res.json())
      .then(setData);
  });
  
  return <div>{data}</div>;
}""",
                "output": """import { useState, useEffect } from 'react';

function DataFetcher() {
  const [data, setData] = useState(null);
  
  useEffect(() => {
    fetch('/api/data')
      .then(res => res.json())
      .then(setData);
  }, []); // Empty dependency array means run once on mount
  
  return <div>{data}</div>;
}"""
            },
            {
                "instruction": "Fix the event handler that loses 'this' context",
                "input": """class Button extends React.Component {
  handleClick() {
    this.setState({ clicked: true });
  }
  
  render() {
    return <button onClick={this.handleClick}>Click</button>;
  }
}""",
                "output": """class Button extends React.Component {
  // Option 1: Arrow function (recommended)
  handleClick = () => {
    this.setState({ clicked: true });
  }
  
  render() {
    return <button onClick={this.handleClick}>Click</button>;
  }
}

// Option 2: Bind in constructor
// constructor(props) {
//   super(props);
//   this.handleClick = this.handleClick.bind(this);
// }"""
            },
            {
                "instruction": "Fix the component that doesn't handle async state updates correctly",
                "input": """function Counter() {
  const [count, setCount] = useState(0);
  
  const increment = () => {
    setCount(count + 1);
    setCount(count + 1);
    setCount(count + 1);
  };
  
  return <button onClick={increment}>Count: {count}</button>;
}""",
                "output": """function Counter() {
  const [count, setCount] = useState(0);
  
  const increment = () => {
    // Use functional update to get previous state
    setCount(prevCount => prevCount + 1);
    setCount(prevCount => prevCount + 1);
    setCount(prevCount => prevCount + 1);
  };
  
  return <button onClick={increment}>Count: {count}</button>;
}"""
            }
        ]
        
        for _ in range(count // len(bug_samples)):
            self.samples.extend(bug_samples)
    
    # ------------------------------------------------------------------------
    # CODE EXPLANATION SAMPLES (20%)
    # ------------------------------------------------------------------------
    
    def generate_explanation_samples(self, count: int = 600):
        """Generate code explanation samples"""
        
        explanations = [
            {
                "instruction": "Explain what the useState hook does in React",
                "input": "",
                "output": """The useState hook is a fundamental React hook that allows you to add state to functional components.

Syntax: const [state, setState] = useState(initialValue);

Key points:
1. Returns an array with two elements: current state and setter function
2. The setter function updates the state and triggers re-render
3. Can be called multiple times to manage multiple state variables
4. State updates are asynchronous

Example:
const [count, setCount] = useState(0);
// count is the current value (initially 0)
// setCount is the function to update it"""
            },
            {
                "instruction": "Explain the useEffect hook and when to use it",
                "input": "",
                "output": """The useEffect hook handles side effects in functional components like data fetching, subscriptions, or manual DOM changes.

Syntax: useEffect(() => { /* effect */ }, [dependencies]);

Key behaviors:
1. Runs after every render by default
2. Dependencies array controls when it runs:
   - No array: runs after every render
   - Empty []: runs once on mount
   - [var1, var2]: runs when dependencies change

3. Return cleanup function to prevent memory leaks:
   useEffect(() => {
     const timer = setInterval(...);
     return () => clearInterval(timer);
   }, []);

Common uses: API calls, event listeners, timers, DOM manipulation"""
            },
            {
                "instruction": "Explain the difference between props and state in React",
                "input": "",
                "output": """Props and state are both data sources in React, but they have different purposes:

PROPS:
- Short for "properties"
- Passed from parent to child component
- Read-only (immutable) in child component
- Used for component configuration
- Example: <Button color="blue" text="Click me" />

STATE:
- Managed within the component
- Can be changed using setState or useState
- Private to the component
- Triggers re-render when updated
- Example: const [isOpen, setIsOpen] = useState(false);

Rule: Props flow down, state stays local. If you need to pass data up, use callback functions via props."""
            },
            {
                "instruction": "Explain React component lifecycle in functional components",
                "input": "",
                "output": """In functional components, lifecycle is managed through hooks:

1. MOUNTING (Component Creation):
   useEffect(() => {
     // Runs once when component mounts
     console.log('Component mounted');
   }, []);

2. UPDATING (When props/state change):
   useEffect(() => {
     // Runs when 'count' changes
     console.log('Count updated');
   }, [count]);

3. UNMOUNTING (Component Removal):
   useEffect(() => {
     return () => {
       // Cleanup function runs on unmount
       console.log('Component will unmount');
     };
   }, []);

This replaces class lifecycle methods like componentDidMount, componentDidUpdate, and componentWillUnmount."""
            },
            {
                "instruction": "Explain what React keys are and why they're important",
                "input": "",
                "output": """Keys help React identify which items in a list have changed, been added, or removed.

Why they matter:
1. Performance: React uses keys to minimize DOM manipulations
2. Component state: Preserves state between re-renders
3. Prevents bugs: Avoids incorrect component reuse

Best practices:
- Use unique, stable IDs: <li key={item.id}>
- Avoid array indices when list can change
- Don't use random values (generates new key each render)

Bad example:
{items.map(item => <li>{item}</li>)} // Missing key

Good example:
{items.map(item => <li key={item.id}>{item.name}</li>)}"""
            }
        ]
        
        for _ in range(count // len(explanations)):
            self.samples.extend(explanations)
    
    # ------------------------------------------------------------------------
    # CONVERSION SAMPLES (15%)
    # ------------------------------------------------------------------------
    
    def generate_conversion_samples(self, count: int = 450):
        """Generate JSX conversion and refactoring samples"""
        
        conversions = [
            {
                "instruction": "Convert this HTML to JSX",
                "input": """<div class="container">
  <label for="username">Username:</label>
  <input type="text" id="username" />
  <button onclick="handleClick()">Submit</button>
</div>""",
                "output": """<div className="container">
  <label htmlFor="username">Username:</label>
  <input type="text" id="username" />
  <button onClick={handleClick}>Submit</button>
</div>

// Key changes:
// - class → className
// - for → htmlFor
// - onclick → onClick (camelCase)
// - String function → function reference"""
            },
            {
                "instruction": "Convert this class component to a functional component with hooks",
                "input": """class Timer extends React.Component {
  constructor(props) {
    super(props);
    this.state = { seconds: 0 };
  }
  
  componentDidMount() {
    this.interval = setInterval(() => {
      this.setState({ seconds: this.state.seconds + 1 });
    }, 1000);
  }
  
  componentWillUnmount() {
    clearInterval(this.interval);
  }
  
  render() {
    return <div>Seconds: {this.state.seconds}</div>;
  }
}""",
                "output": """import { useState, useEffect } from 'react';

function Timer() {
  const [seconds, setSeconds] = useState(0);
  
  useEffect(() => {
    const interval = setInterval(() => {
      setSeconds(prevSeconds => prevSeconds + 1);
    }, 1000);
    
    return () => clearInterval(interval);
  }, []);
  
  return <div>Seconds: {seconds}</div>;
}

export default Timer;"""
            },
            {
                "instruction": "Refactor this component to use custom hooks",
                "input": """function UserProfile() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    fetch('/api/user')
      .then(res => res.json())
      .then(data => {
        setUser(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err);
        setLoading(false);
      });
  }, []);
  
  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;
  return <div>{user.name}</div>;
}""",
                "output": """import { useState, useEffect } from 'react';

// Custom hook
function useFetch(url) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    fetch(url)
      .then(res => res.json())
      .then(data => {
        setData(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err);
        setLoading(false);
      });
  }, [url]);
  
  return { data, loading, error };
}

// Component using custom hook
function UserProfile() {
  const { data: user, loading, error } = useFetch('/api/user');
  
  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;
  return <div>{user.name}</div>;
}"""
            }
        ]
        
        for _ in range(count // len(conversions)):
            self.samples.extend(conversions)
    
    # ------------------------------------------------------------------------
    # GENERATE AND SAVE
    # ------------------------------------------------------------------------
    
    def generate_all(self):
        """Generate complete dataset"""
        print("🔨 Generating React training dataset...")
        print("  📦 Component generation samples...")
        self.generate_component_samples(1200)
        print("  🐛 Bug fixing samples...")
        self.generate_bug_fixing_samples(750)
        print("  📚 Code explanation samples...")
        self.generate_explanation_samples(600)
        print("  🔄 Conversion samples...")
        self.generate_conversion_samples(450)
        
        # Shuffle for better training
        random.shuffle(self.samples)
        print(f"\n✅ Generated {len(self.samples)} total samples")
        return self.samples
    
    def save_to_jsonl(self, filename: str = "react_train.jsonl"):
        """Save dataset to JSONL file"""
        if not self.samples:
            self.generate_all()
        
        output_path = Path(filename)
        with output_path.open('w', encoding='utf-8') as f:
            for sample in self.samples:
                f.write(json.dumps(sample) + '\n')
        
        print(f"💾 Dataset saved to {filename}")
        print(f"   Total samples: {len(self.samples)}")
        
        # Show distribution
        print("\n📊 Dataset Distribution:")
        print(f"   Component Generation: ~{len([s for s in self.samples if 'create' in s['instruction'].lower() or 'build' in s['instruction'].lower()])} samples")
        print(f"   Bug Fixing: ~{len([s for s in self.samples if 'fix' in s['instruction'].lower()])} samples")
        print(f"   Explanations: ~{len([s for s in self.samples if 'explain' in s['instruction'].lower()])} samples")
        print(f"   Conversions: ~{len([s for s in self.samples if 'convert' in s['instruction'].lower()])} samples")
        
        return output_path


# ============================================================================
# EXISTING DATASET SOURCES
# ============================================================================

def download_existing_datasets():
    """Guide to find existing React datasets"""
    
    print("\n" + "="*70)
    print("📥 EXISTING REACT DATASETS YOU CAN USE")
    print("="*70)
    
    datasets = [
        {
            "name": "Hugging Face - Code Alpaca",
            "url": "https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k",
            "description": "20k code instruction samples, includes React",
            "command": 'from datasets import load_dataset\nds = load_dataset("sahil2801/CodeAlpaca-20k")'
        },
        {
            "name": "Hugging Face - Code Instructions",
            "url": "https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca",
            "description": "Filter for JavaScript/React examples",
            "command": 'from datasets import load_dataset\nds = load_dataset("iamtarun/python_code_instructions_18k_alpaca")'
        },
        {
            "name": "GitHub - React Code Samples",
            "url": "https://github.com/search?q=react+components+examples",
            "description": "Scrape real React components from GitHub repos",
            "command": "Use GitHub API to download React component files"
        },
        {
            "name": "Stack Overflow",
            "url": "https://stackoverflow.com/questions/tagged/reactjs",
            "description": "Q&A pairs about React (use Stack Exchange API)",
            "command": "Use StackAPI to fetch React questions and answers"
        }
    ]
    
    for i, ds in enumerate(datasets, 1):
        print(f"\n{i}. {ds['name']}")
        print(f"   URL: {ds['url']}")
        print(f"   Description: {ds['description']}")
        print(f"   Command: {ds['command']}")
    
    print("\n" + "="*70)


# ============================================================================
# DATASET ENHANCEMENT
# ============================================================================

def enhance_dataset_with_variations(input_file: str, output_file: str):
    """Add variations to existing dataset to increase size"""
    print(f"\n🔧 Enhancing dataset: {input_file}")
    
    samples = []
    with open(input_file, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    enhanced = []
    for sample in samples:
        # Original sample
        enhanced.append(sample)
        
        # Create variations by modifying instruction
        instruction = sample['instruction']
        output = sample['output']
        
        variations = [
            instruction.replace('Create', 'Build'),
            instruction.replace('Build', 'Develop'),
            f"Write a React component that {instruction.lower().replace('create a react ', '').replace('build a react ', '')}",
        ]
        
        for var_instruction in variations[:2]:  # Add 2 variations per sample
            if var_instruction != instruction:
                enhanced.append({
                    "instruction": var_instruction,
                    "input": sample.get('input', ''),
                    "output": output
                })
    
    # Save enhanced dataset
    with open(output_file, 'w') as f:
        for sample in enhanced:
            f.write(json.dumps(sample) + '\n')
    
    print(f"✅ Enhanced dataset saved to {output_file}")
    print(f"   Original: {len(samples)} samples")
    print(f"   Enhanced: {len(enhanced)} samples")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("REACT TRAINING DATASET GENERATOR")
    print("=" * 70)
    
    # Option 1: Generate new dataset
    print("\n🎯 OPTION 1: Generate Fresh Dataset")
    generator = ReactDatasetGenerator()
    generator.generate_all()
    generator.save_to_jsonl("react_train.jsonl")
    
    # Option 2: Show existing dataset sources
    download_existing_datasets()
    
    # Option 3: Enhance existing dataset
    print("\n\n🎯 OPTION 3: Enhance Existing Dataset")
    print("Run: enhance_dataset_with_variations('react_train.jsonl', 'react_train_enhanced.jsonl')")
    
    print("\n" + "="*70)
    print("✅ ALL DONE!")
    print("="*70)
    print("\nYour dataset is ready at: react_train.jsonl")
    print("Use this file with the quantum training script!")
