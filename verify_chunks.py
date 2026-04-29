import json

# Load the new chunks
with open("data/processed/chunks.json", "r") as f:
    chunks = json.load(f)

print(f"✓ Total chunks: {len(chunks)}")
print(f"\nFirst chunk sample (first 500 chars):")
print(chunks[0][:500])
print("\n" + "="*80)

# Search for candidate names in chunks
candidates_to_find = ["Nana Akufo Addo", "John Dramani Mahama", "Ivor Kobina Greenstreet"]
print("\n🔍 SEARCHING FOR KEY CANDIDATES:")

for candidate in candidates_to_find:
    found = False
    for i, chunk in enumerate(chunks):
        if candidate in chunk:
            found = True
            # Find the part of the chunk with vote info
            start = chunk.find(candidate) - 100
            end = chunk.find(candidate) + 300
            print(f"\n✓ {candidate} found in chunk {i}:")
            print(f"  ...{chunk[max(0,start):end]}...")
            break
    if not found:
        print(f"\n✗ {candidate} NOT FOUND in chunks")

# Show stats
print("\n" + "="*80)
print("📊 CHUNK CONTENT QUALITY:")
print(f"- Chunks containing 'candidate': {sum(1 for c in chunks if 'candidate:' in c)}")
print(f"- Chunks containing 'votes': {sum(1 for c in chunks if 'votes:' in c)}")
print(f"- Chunks containing 'region': {sum(1 for c in chunks if 'region' in c.lower())}")
print(f"- Chunks with only numbers (BAD): {sum(1 for c in chunks if c.replace(' ', '').replace('.', '').isdigit())}")
