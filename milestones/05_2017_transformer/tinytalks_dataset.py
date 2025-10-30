"""
TinyTalks Dataset - Small Conversational Dataset for Transformer Training
==========================================================================

A carefully curated micro-dataset for training a chatbot in 10-15 minutes.

Design Principles:
- Simple Q&A format
- Short responses (< 50 characters)
- Common patterns repeated for learning
- Character-level tokenization friendly
- Covers basic conversational topics
"""

def create_tinytalks_dataset():
    """
    Create a small but diverse conversational dataset.
    
    Categories:
    1. Greetings & Politeness
    2. Simple Facts
    3. Yes/No Questions
    4. Weather & Time
    5. Feelings & States
    6. Simple Math
    7. Colors & Objects
    8. Names & Identity
    """
    
    conversations = [
        # === GREETINGS (repeated 5x for strong learning) ===
        ("Hi", "Hello! How can I help you?"),
        ("Hello", "Hi there! How are you?"),
        ("Good morning", "Good morning! Nice day!"),
        ("How are you", "I am doing well, thanks!"),
        ("Bye", "Goodbye! Have a great day!"),
        
        # Repeat for learning
        ("Hi", "Hello! How can I help you?"),
        ("Hello", "Hi there! How are you?"),
        ("How are you", "I am doing well, thanks!"),
        ("Bye", "Goodbye! Have a great day!"),
        
        ("Hi", "Hello! How can I help you?"),
        ("Hello", "Hi there! How are you?"),
        ("Bye", "Goodbye! Have a great day!"),
        
        # === SIMPLE FACTS (repeated 3x) ===
        ("What is the sky", "The sky is blue"),
        ("What is grass", "Grass is green"),
        ("What is the sun", "The sun is bright and yellow"),
        ("What is water", "Water is wet and clear"),
        ("What is fire", "Fire is hot and red"),
        
        ("What is the sky", "The sky is blue"),
        ("What is grass", "Grass is green"),
        ("What is water", "Water is wet and clear"),
        
        ("What is the sky", "The sky is blue"),
        ("What is grass", "Grass is green"),
        
        # === YES/NO QUESTIONS (repeated 3x) ===
        ("Is the sky blue", "Yes, the sky is blue"),
        ("Is grass green", "Yes, grass is green"),
        ("Is the sun cold", "No, the sun is hot"),
        ("Is water dry", "No, water is wet"),
        ("Is fire safe", "No, fire is dangerous"),
        
        ("Is the sky blue", "Yes, the sky is blue"),
        ("Is grass green", "Yes, grass is green"),
        ("Is the sun cold", "No, the sun is hot"),
        
        ("Is the sky blue", "Yes, the sky is blue"),
        ("Is grass green", "Yes, grass is green"),
        
        # === WEATHER (repeated 3x) ===
        ("Is it sunny", "Yes, it is sunny today"),
        ("Is it raining", "No, it is not raining"),
        ("Is it cold", "No, it is warm today"),
        ("What is the weather", "The weather is nice"),
        
        ("Is it sunny", "Yes, it is sunny today"),
        ("What is the weather", "The weather is nice"),
        
        ("Is it sunny", "Yes, it is sunny today"),
        
        # === FEELINGS (repeated 3x) ===
        ("Are you happy", "Yes, I am happy"),
        ("Are you sad", "No, I am not sad"),
        ("Are you tired", "No, I feel good"),
        ("Do you like learning", "Yes, I love learning"),
        
        ("Are you happy", "Yes, I am happy"),
        ("Do you like learning", "Yes, I love learning"),
        
        ("Are you happy", "Yes, I am happy"),
        
        # === SIMPLE MATH (repeated 3x) ===
        ("What is 1 plus 1", "1 plus 1 equals 2"),
        ("What is 2 plus 2", "2 plus 2 equals 4"),
        ("What is 3 plus 3", "3 plus 3 equals 6"),
        ("What is 5 plus 5", "5 plus 5 equals 10"),
        
        ("What is 1 plus 1", "1 plus 1 equals 2"),
        ("What is 2 plus 2", "2 plus 2 equals 4"),
        
        ("What is 1 plus 1", "1 plus 1 equals 2"),
        
        # === COLORS (repeated 3x) ===
        ("What color is the sky", "The sky is blue"),
        ("What color is grass", "Grass is green"),
        ("What color is the sun", "The sun is yellow"),
        ("What color is snow", "Snow is white"),
        
        ("What color is the sky", "The sky is blue"),
        ("What color is grass", "Grass is green"),
        
        ("What color is the sky", "The sky is blue"),
        
        # === IDENTITY (repeated 3x) ===
        ("What is your name", "I am TinyBot"),
        ("Who are you", "I am TinyBot, your helper"),
        ("What do you do", "I help answer questions"),
        
        ("What is your name", "I am TinyBot"),
        ("Who are you", "I am TinyBot, your helper"),
        
        ("What is your name", "I am TinyBot"),
        
        # === CAPABILITIES (repeated 2x) ===
        ("Can you help me", "Yes, I can help you"),
        ("Can you talk", "Yes, I can talk with you"),
        ("Do you understand", "Yes, I understand you"),
        
        ("Can you help me", "Yes, I can help you"),
        ("Can you talk", "Yes, I can talk with you"),
    ]
    
    return conversations


def get_dataset_stats():
    """Get statistics about the dataset."""
    conversations = create_tinytalks_dataset()
    
    unique_conversations = set(conversations)
    total_chars = sum(len(q) + len(a) for q, a in conversations)
    avg_question_len = sum(len(q) for q, _ in conversations) / len(conversations)
    avg_answer_len = sum(len(a) for _, a in conversations) / len(conversations)
    
    return {
        'total_examples': len(conversations),
        'unique_examples': len(unique_conversations),
        'repetition_factor': len(conversations) / len(unique_conversations),
        'total_chars': total_chars,
        'avg_question_len': avg_question_len,
        'avg_answer_len': avg_answer_len,
        'categories': [
            'Greetings (5x repeat)',
            'Simple Facts (3x repeat)',
            'Yes/No Questions (3x repeat)',
            'Weather (3x repeat)',
            'Feelings (3x repeat)',
            'Simple Math (3x repeat)',
            'Colors (3x repeat)',
            'Identity (3x repeat)',
            'Capabilities (2x repeat)'
        ]
    }


def print_dataset_info():
    """Print dataset information."""
    conversations = create_tinytalks_dataset()
    stats = get_dataset_stats()
    
    print("=" * 70)
    print("TINYTALKS DATASET")
    print("=" * 70)
    print()
    print(f"Total examples: {stats['total_examples']}")
    print(f"Unique examples: {stats['unique_examples']}")
    print(f"Repetition factor: {stats['repetition_factor']:.1f}x")
    print(f"Average question length: {stats['avg_question_len']:.1f} chars")
    print(f"Average answer length: {stats['avg_answer_len']:.1f} chars")
    print()
    print("Categories:")
    for cat in stats['categories']:
        print(f"  • {cat}")
    print()
    print("Sample conversations:")
    print("-" * 70)
    
    # Show 10 random unique examples
    unique = list(set(conversations))
    import random
    random.seed(42)
    samples = random.sample(unique, min(10, len(unique)))
    
    for q, a in samples:
        print(f"Q: {q}")
        print(f"A: {a}")
        print()


if __name__ == "__main__":
    print_dataset_info()

