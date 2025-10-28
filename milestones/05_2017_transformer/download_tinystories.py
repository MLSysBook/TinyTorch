#!/usr/bin/env python3
"""
Download and prepare TinyStories dataset for TinyTorch training.

TinyStories is a dataset of simple, synthetic stories designed for
training small language models. It's much easier than Shakespeare!
"""

import os
import urllib.request

def download_tinystories():
    """Download TinyStories dataset."""
    
    # Create data directory
    data_dir = os.path.join(os.path.dirname(__file__), '../datasets/tinystories')
    os.makedirs(data_dir, exist_ok=True)
    
    # TinyStories validation set (smaller, good for testing)
    urls = {
        'tiny_val': 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt',
        'tiny_train_small': 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt'
    }
    
    print("📥 Downloading TinyStories dataset...")
    print("="*70)
    
    # Start with validation set (much smaller for testing)
    filename = 'tinystories_val.txt'
    filepath = os.path.join(data_dir, filename)
    
    if os.path.exists(filepath):
        print(f"✅ {filename} already exists")
        size = os.path.getsize(filepath) / (1024 * 1024)
        print(f"   Size: {size:.2f} MB")
    else:
        print(f"⬇️  Downloading {filename}...")
        try:
            urllib.request.urlretrieve(urls['tiny_val'], filepath)
            size = os.path.getsize(filepath) / (1024 * 1024)
            print(f"✅ Downloaded! Size: {size:.2f} MB")
        except Exception as e:
            print(f"❌ Error downloading: {e}")
            print("\n💡 Alternative: Download manually from:")
            print(f"   {urls['tiny_val']}")
            print(f"   Save to: {filepath}")
            return None
    
    # Read and show sample
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"\n📊 Dataset Stats:")
    print(f"   Total characters: {len(text):,}")
    print(f"   Total words: {len(text.split()):,}")
    print(f"   Unique characters: {len(set(text))}")
    
    # Show first story
    stories = text.split('<|endoftext|>')
    if len(stories) > 0:
        first_story = stories[0].strip()
        print(f"\n📖 Sample Story:")
        print("   " + "-"*66)
        print("   " + first_story[:300].replace('\n', '\n   '))
        if len(first_story) > 300:
            print("   ...")
        print("   " + "-"*66)
    
    print(f"\n✅ TinyStories ready for training!")
    print(f"   Location: {filepath}")
    
    return filepath

if __name__ == '__main__':
    download_tinystories()
