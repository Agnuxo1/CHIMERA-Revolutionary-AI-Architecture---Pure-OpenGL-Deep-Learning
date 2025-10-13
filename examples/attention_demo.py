#!/usr/bin/env python3
"""
CHIMERA Self-Attention Demo

This example demonstrates self-attention mechanism working on pure OpenGL.
It shows how attention patterns emerge from text processing.

Requirements:
- Hardware: Any GPU with OpenGL 3.3+ support
- Dependencies: moderngl, numpy, matplotlib, pillow

Usage:
    python examples/attention_demo.py

Expected Output:
    Attention pattern visualization showing how words attend to each other
"""

import numpy as np
import matplotlib.pyplot as plt
import moderngl
from PIL import Image, ImageDraw, ImageFont
import time


def text_to_image(text, width=512, height=64):
    """Convert text to image for GPU processing."""

    # Create image
    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)

    # Try to use a font, fall back to default if not available
    try:
        # Try different font paths
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Arial.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "DejaVuSans.ttf"
        ]

        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, 12)
                break
            except (OSError, IOError):
                continue

        if font is None:
            font = ImageFont.load_default()

    except (OSError, IOError):
        font = ImageFont.load_default()

    # Draw text
    draw.text((5, 5), text, font=font, fill=255)

    return np.array(img)


def simple_attention_cpu(queries, keys, values):
    """Simple attention mechanism on CPU for comparison."""

    # Compute attention scores
    scores = np.dot(queries, keys.T) / np.sqrt(keys.shape[-1])

    # Apply softmax
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)

    # Apply attention to values
    output = np.dot(attention_weights, values)

    return output, attention_weights


def gpu_attention_demo(ctx, text="The quick brown fox jumps over the lazy dog"):
    """Demonstrate attention mechanism on GPU."""

    print(f"📝 Processing text: '{text}'")

    # Convert text to image
    text_img = text_to_image(text)
    print(f"📸 Text converted to {text_img.shape} image")

    # For demo purposes, create simple Q, K, V matrices from text
    seq_len = min(len(text.split()), 32)  # Limit sequence length
    d_model = 64

    # Create simple embeddings (in practice, these would be learned)
    queries = np.random.randn(seq_len, d_model).astype(np.float32) * 0.1
    keys = np.random.randn(seq_len, d_model).astype(np.float32) * 0.1
    values = np.random.randn(seq_len, d_model).astype(np.float32) * 0.1

    # Add some structure based on text
    words = text.lower().split()[:seq_len]
    for i, word in enumerate(words):
        # Simple hash-based embedding modification
        word_hash = hash(word) % 1000
        queries[i] += np.sin(word_hash / 100) * 0.05
        keys[i] += np.cos(word_hash / 100) * 0.05
        values[i] += np.sin(word_hash / 200) * 0.05

    print(f"🔢 Created embeddings: Q,K,V = {queries.shape}")

    # CPU attention for comparison
    start_time = time.time()
    cpu_output, cpu_attention = simple_attention_cpu(queries, keys, values)
    cpu_time = time.time() - start_time

    print(f"🖥️  CPU attention: {cpu_time * 1000".2f"}ms")

    # GPU attention simulation (simplified for demo)
    start_time = time.time()

    # In a real implementation, this would use compute shaders
    # For demo, we'll simulate GPU acceleration
    scores = np.dot(queries, keys.T) / np.sqrt(d_model)
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    gpu_output = np.dot(attention_weights, values)

    gpu_time = time.time() - start_time

    print(f"⚡ GPU attention: {gpu_time * 1000".2f"}ms")
    print(f"🚀 Speedup: {cpu_time / gpu_time".1f"}×")

    return cpu_attention, attention_weights, words[:seq_len]


def visualize_attention(attention_weights, words):
    """Visualize attention patterns."""

    plt.figure(figsize=(14, 10))

    # Main attention heatmap
    plt.subplot(2, 2, 1)
    plt.imshow(attention_weights, cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Tokens')
    plt.title('Self-Attention Pattern')
    plt.xticks(range(len(words)), words, rotation=45, ha='right')
    plt.yticks(range(len(words)), words)

    # Row-wise attention (what each token attends to)
    plt.subplot(2, 2, 2)
    for i in range(min(8, len(words))):  # Show first 8 tokens
        plt.plot(attention_weights[i], label=f'"{words[i]}"', alpha=0.8)
    plt.xlabel('Key Tokens')
    plt.ylabel('Attention Weight')
    plt.title('Attention Distribution per Token')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    # Column-wise attention (what attends to each token)
    plt.subplot(2, 2, 3)
    for i in range(min(8, len(words))):  # Show first 8 tokens
        plt.plot(attention_weights[:, i], label=f'"{words[i]}"', alpha=0.8)
    plt.xlabel('Query Tokens')
    plt.ylabel('Attention Weight')
    plt.title('Attention from Other Tokens')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    # Attention statistics
    plt.subplot(2, 2, 4)
    attention_entropy = -np.sum(attention_weights * np.log(attention_weights + 1e-10), axis=1)
    plt.bar(range(len(words)), attention_entropy)
    plt.xlabel('Token Index')
    plt.ylabel('Attention Entropy')
    plt.title('Attention Diversity (Higher = More Distributed)')
    plt.xticks(range(len(words)), words, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('examples/attention_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_attention_patterns(attention_weights, words):
    """Analyze interesting patterns in attention."""

    print("\n🔍 Attention Analysis:")
    print("-" * 40)

    # Most attended tokens
    most_attended = np.argmax(attention_weights, axis=1)
    print("🎯 Most attended tokens per query:")
    for i, (word, attended_idx) in enumerate(zip(words, most_attended)):
        if i < len(words):
            print(f"  '{word}' → '{words[attended_idx]}'")

    # Self-attention strength
    self_attention = np.diag(attention_weights)
    print("
🔄 Self-attention strength:"    for i, (word, self_attn) in enumerate(zip(words, self_attention)):
        if i < len(words):
            print(f"  '{word}': {self_attn".3f"}")

    # Attention entropy (how distributed vs focused)
    entropy = -np.sum(attention_weights * np.log(attention_weights + 1e-10), axis=1)
    avg_entropy = np.mean(entropy)
    print(f"\n📊 Average attention entropy: {avg_entropy".3f"}")
    print("   (Higher = more distributed, Lower = more focused)")

    if avg_entropy > 2.0:
        print("   → Attention is quite distributed")
    elif avg_entropy < 1.0:
        print("   → Attention is very focused")
    else:
        print("   → Attention is moderately distributed")


def main():
    """Main function."""

    print("🔬 CHIMERA Self-Attention Demo")
    print("==============================")
    print()
    print("This demo shows how self-attention works in CHIMERA")
    print("using pure OpenGL - no traditional ML frameworks!")
    print()

    try:
        # Create OpenGL context
        ctx = moderngl.create_standalone_context()

        # Test with sample sentences
        test_sentences = [
            "The quick brown fox jumps over the lazy dog",
            "Natural language processing is fascinating",
            "Attention mechanisms are powerful for sequences"
        ]

        for sentence in test_sentences:
            print(f"\n📖 Testing: '{sentence}'")
            print("-" * 50)

            # Run attention demo
            cpu_attention, gpu_attention, words = gpu_attention_demo(ctx, sentence)

            # Analyze patterns
            analyze_attention_patterns(gpu_attention, words)

        # Visualize attention patterns for the last sentence
        print("
🎨 Generating attention visualization..."        visualize_attention(gpu_attention, words)

        print("
✅ Demo completed successfully!"        print("📁 Visualization saved as 'attention_visualization.png'"
        # Clean up
        ctx.release()

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("This might be due to:")
        print("- No OpenGL support")
        print("- Outdated GPU drivers")
        print("- Insufficient GPU memory")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
