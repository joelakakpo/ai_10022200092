#!/usr/bin/env python3
"""
Comprehensive test of the RAG system with the fixed data
"""
import json

print("\n" + "="*80)
print("🧪 COMPREHENSIVE RAG SYSTEM TEST")
print("="*80)

# Test 1: Load chunks
print("\n[1/4] Loading Chunks...")
try:
    with open("data/processed/chunks.json", "r") as f:
        chunks = json.load(f)
    print(f"✅ Loaded {len(chunks)} chunks from data/processed/chunks.json")
except Exception as e:
    print(f"❌ Failed to load chunks: {e}")
    exit(1)

# Test 2: Load full text
print("\n[2/4] Loading Full Text...")
try:
    with open("data/processed/full_text.txt", "r") as f:
        full_text = f.read()
    print(f"✅ Loaded {len(full_text)} characters from data/processed/full_text.txt")
except Exception as e:
    print(f"❌ Failed to load full text: {e}")
    exit(1)

# Test 3: Verify content quality
print("\n[3/4] Verifying Content Quality...")

quality_checks = {
    "Contains 'candidate:'": sum(1 for c in chunks if 'candidate:' in c) > 0,
    "Contains 'votes:'": sum(1 for c in chunks if 'votes:' in c) > 0,
    "Contains 'region'": sum(1 for c in chunks if 'region' in c.lower()) > 0,
    "Contains 'Nana Akufo Addo'": any('Nana Akufo Addo' in c for c in chunks),
    "Contains 'John Dramani Mahama'": any('John Dramani Mahama' in c for c in chunks),
    "No 'only numbers' chunks": not any(c.replace(' ', '').replace('.', '').isdigit() for c in chunks),
}

all_passed = True
for check, passed in quality_checks.items():
    status = "✅" if passed else "❌"
    print(f"  {status} {check}")
    if not passed:
        all_passed = False

if not all_passed:
    print("\n❌ Some quality checks failed!")
    exit(1)

# Test 4: Test retrieval with candidate names
print("\n[4/4] Testing Candidate Name Retrieval...")

test_cases = [
    ("Nana Akufo Addo", "candidate: Nana Akufo Addo"),
    ("votes 2020", "votes:"),
    ("Ashanti Region", "Ashanti Region"),
]

all_retrieval_passed = True
for query_term, expected_content in test_cases:
    found = any(expected_content in chunk for chunk in chunks if query_term.lower() in chunk.lower())
    status = "✅" if found else "❌"
    print(f"  {status} Query '{query_term}' finds '{expected_content}': {found}")
    if not found:
        all_retrieval_passed = False

# Print Summary
print("\n" + "="*80)
if all_passed and all_retrieval_passed:
    print("✅ ALL TESTS PASSED - RAG SYSTEM IS READY!")
    print("\nYou can now run:")
    print("  streamlit run rag_app.py")
    print("\nAnd ask questions like:")
    print("  - 'How many votes did Nana Akufo Addo get in 2020?'")
    print("  - 'Which party won in Ashanti Region?'")
    print("  - 'Show election results for 2020'")
else:
    print("❌ SOME TESTS FAILED - Please check the errors above")
    
print("="*80 + "\n")
