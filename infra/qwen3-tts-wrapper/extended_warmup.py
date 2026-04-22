#!/usr/bin/env python3
"""
Extended warmup and benchmark to verify torch.compile() speedup.
"""

import time
import requests

API_URL = "http://localhost:8880/v1/audio/speech"

def warmup_and_benchmark():
    print("🔥 Extended Warmup for torch.compile()")
    print("=" * 60)
    
    # More warmup requests to fully compile the model
    warmup_texts = [
        "Short warmup test one.",
        "This is the second warmup request for compilation.",
        "Third warmup request to ensure torch compile is fully optimized.",
        "Fourth warmup with a slightly longer text for better coverage.",
        "Fifth and final warmup to maximize compilation benefits.",
    ]
    
    for i, text in enumerate(warmup_texts, 1):
        print(f"Warmup {i}/5...", end=" ")
        start = time.time()
        try:
            response = requests.post(
                API_URL,
                json={"input": text, "voice": "Vivian", "model": "tts-1"},
                timeout=60
            )
            elapsed = time.time() - start
            print(f"{elapsed:.2f}s ({'✓' if response.status_code == 200 else '✗'})")
        except Exception as e:
            print(f"Error: {e}")
    
    # Now benchmark
    print("\n📊 Post-Warmup Benchmark")
    print("=" * 60)
    
    test_cases = [
        "Hello world test.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the way we live and work with amazing new capabilities.",
    ]
    
    for text in test_cases:
        words = len(text.split())
        times = []
        
        # Run each test 3 times
        for run in range(3):
            start = time.time()
            response = requests.post(
                API_URL,
                json={"input": text, "voice": "Vivian", "model": "tts-1"},
                timeout=60
            )
            elapsed = time.time() - start
            if response.status_code == 200:
                times.append(elapsed)
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            audio_dur = words * 0.6
            rtf = min_time / audio_dur
            print(f"{words:2d} words: {min_time:.2f}s (RTF {rtf:.2f})")
    
    print("=" * 60)

if __name__ == "__main__":
    warmup_and_benchmark()
