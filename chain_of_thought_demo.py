"""
DEMO: Chain-of-Thought Orchestration for RAG

This demo compares:
1. Standard Retrieval (baseline)
2. Chain-of-Thought Reasoning (novel feature)

Shows the improvement in accuracy and transparency for complex queries.
"""

import json
from retrieval_pipeline.embeddings import embed_texts, embed_query
from retrieval_pipeline.storage import build_index
from retrieval_pipeline.retrieval import search
from retrieval_pipeline.chain_of_thought import (
    chain_of_thought_retrieval_augmented_generation,
    decompose_query_into_steps
)
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# ============================================================================
# SAMPLE DOCUMENTS (Knowledge Base)
# ============================================================================

DOCUMENTS = [
    """Apple Inc. was founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne. 
    Apple revolutionized personal computing with the Apple II and later the Macintosh. 
    The company's focus on design and user experience made it a leader in consumer electronics.""",
    
    """Microsoft was founded in 1975 by Bill Gates and Paul Allen. 
    Microsoft became dominant through its MS-DOS operating system and later Windows. 
    The company's enterprise software strategy made it one of the world's most valuable companies.""",
    
    """The personal computer revolution of the 1980s transformed society. 
    Key players included Apple, IBM, Commodore, and Atari. 
    This era saw the shift from mainframes to consumer-accessible computing.""",
    
    """Steve Jobs left Apple in 1985 but founded NeXT Computer Company, 
    which developed innovative workstations. Later, Jobs acquired Pixar and became its majority shareholder. 
    Jobs returned to Apple in 1997 and transformed the company with the iMac, iPod, and iPhone.""",
    
    """The iPhone, released in 2007, revolutionized mobile computing and became the best-selling phone of all time. 
    It introduced touch-screen interfaces to mainstream consumers. 
    The App Store ecosystem created an entirely new software market.""",
    
    """Android, developed by Google, became the most used mobile operating system globally. 
    It competes directly with Apple's iOS. Many manufacturers including Samsung, Xiaomi, and OnePlus use Android.""",
]

# ============================================================================
# COMPARISON FUNCTION
# ============================================================================

def simple_rag_answer(query: str, index, texts: list) -> dict:
    """Standard RAG: simple retrieval + LLM."""
    
    # Retrieve
    query_emb = embed_query(query)
    results = search(index, query_emb, texts, k=3)
    context = "\n".join([text for text, _ in results])
    
    # Generate answer
    prompt = f"""Answer this question using ONLY the provided context:

Context:
{context}

Question: {query}

Answer:"""
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    
    answer = response.choices[0].message.content
    
    return {
        'method': 'Simple RAG',
        'query': query,
        'retrieved_docs_count': len(results),
        'answer': answer,
        'reasoning_transparency': 'Low (single retrieval pass)',
        'interpretation': 'Direct retrieval + answer'
    }


def chain_of_thought_answer(query: str, index, texts: list) -> dict:
    """Chain-of-Thought RAG: decompose → multi-step retrieve → reason → synthesize."""
    
    result = chain_of_thought_retrieval_augmented_generation(
        query=query,
        index=index,
        texts=texts,
        model="llama-3.3-70b-versatile",
        verbose=False  # Disable verbose for cleaner output
    )
    
    return {
        'method': 'Chain-of-Thought RAG',
        'query': query,
        'complexity': result['reasoning_type'],
        'steps': result['step_count'],
        'answer': result['final_answer'],
        'confidence': result['overall_confidence'],
        'reasoning_transparency': 'High (multi-step with evidence)',
        'evidence': result['evidence_summary'],
        'consistency_notes': result['consistency_notes'],
        'full_chain': result['transparency_report']
    }

# ============================================================================
# MAIN DEMO
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CHAIN-OF-THOUGHT ORCHESTRATION DEMO FOR RAG")
    print("=" * 80)
    
    # Build index
    print("\n📚 Building knowledge base index...")
    embeddings = embed_texts(DOCUMENTS)
    index = build_index(embeddings)
    print(f"   ✅ {len(DOCUMENTS)} documents indexed")
    
    # Test queries: progressively complex
    test_queries = [
        {
            'question': "Who founded Apple?",
            'type': 'Simple (factual)',
            'expected_complexity': 'single_hop'
        },
        {
            'question': "What happened to Steve Jobs and how did it affect Apple?",
            'type': 'Medium (causal)',
            'expected_complexity': 'multi_hop'
        },
        {
            'question': "How did the personal computer revolution influence the mobile computing industry, particularly through companies like Apple?",
            'type': 'Complex (causal + comparative)',
            'expected_complexity': 'multi_hop'
        }
    ]
    
    results_log = []
    
    for test_case in test_queries:
        print(f"\n{'='*80}")
        print(f"QUERY ({test_case['type']}): {test_case['question']}")
        print(f"Expected Complexity: {test_case['expected_complexity']}")
        print(f"{'='*80}")
        
        # Method 1: Simple RAG
        print("\n--- METHOD 1: SIMPLE RAG ---")
        simple_result = simple_rag_answer(test_case['question'], index, DOCUMENTS)
        print(f"Answer: {simple_result['answer'][:200]}...")
        print(f"Reasoning Transparency: {simple_result['reasoning_transparency']}")
        
        # Method 2: Chain-of-Thought RAG
        print("\n--- METHOD 2: CHAIN-OF-THOUGHT RAG ---")
        cot_result = chain_of_thought_answer(test_case['question'], index, DOCUMENTS)
        print(f"Reasoning Type: {cot_result['complexity']}")
        print(f"Number of Reasoning Steps: {cot_result['steps']}")
        print(f"Overall Confidence: {cot_result['confidence']:.2f}")
        print(f"Answer: {cot_result['answer'][:200]}...")
        print(f"Evidence: {cot_result['evidence'][:150]}...")
        print(f"Consistency: {cot_result['consistency_notes']}")
        print(f"Reasoning Transparency: {cot_result['reasoning_transparency']}")
        
        # Save comparison
        results_log.append({
            'query': test_case['question'],
            'type': test_case['type'],
            'simple_rag': {
                'answer': simple_result['answer'],
                'transparency': simple_result['reasoning_transparency']
            },
            'chain_of_thought': {
                'reasoning_type': cot_result['complexity'],
                'steps': cot_result['steps'],
                'confidence': cot_result['confidence'],
                'answer': cot_result['answer'],
                'evidence': cot_result['evidence'],
                'transparency': cot_result['reasoning_transparency']
            }
        })
        
        # Print full reasoning chain for first complex query
        if cot_result['steps'] > 1:
            print("\n🔗 FULL TRANSPARENCY REPORT:")
            print(cot_result['full_chain'][:500] + "...\n")
    
    # Save detailed results
    print(f"\n{'='*80}")
    print("SAVING RESULTS...")
    with open('chain_of_thought_comparison.json', 'w') as f:
        json.dump(results_log, f, indent=2)
    print("✅ Results saved to: chain_of_thought_comparison.json")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY: CHAIN-OF-THOUGHT RAG ADVANTAGES")
    print(f"{'='*80}")
    print("""
✅ IMPROVEMENTS OVER STANDARD RAG:

1. COMPLEX QUERY HANDLING
   - Decomposes complex questions into logical sub-questions
   - Retrieves context for each reasoning step
   - Example: "How did X affect Y?" broken into:
     * What is X? 
     * What is Y?
     * What is the relationship between X and Y?

2. TRANSPARENCY & INTERPRETABILITY
   - Shows full reasoning chain
   - Cites evidence for each conclusion
   - Identifies consistency/conflicts between steps
   - Ideal for high-stakes applications (legal, medical, etc.)

3. MULTI-HOP REASONING
   - Handles questions requiring multiple document hops
   - Accumulates context across reasoning steps
   - Reduces hallucination through evidence tracking

4. CONFIDENCE SCORING
   - Retrieval confidence per step
   - Reasoning confidence per conclusion
   - Overall confidence in final answer
   - Enables uncertainty quantification

5. NOVEL CAPABILITIES
   - Comparative analysis (comparing entities/concepts)
   - Causal reasoning (understanding cause-effect relationships)
   - Aggregation (combining multiple pieces of information)
   - Dependency tracking (which steps depend on which)

📊 WHEN TO USE CHAIN-OF-THOUGHT:
   ✅ Complex multi-part questions
   ✅ Need explainability
   ✅ High-stakes decisions
   ✅ Fact-checking requirements
   ✅ Research or investigation tasks

⚡ WHEN TO USE SIMPLE RAG:
   ✅ Simple factual queries
   ✅ Real-time requirements
   ✅ Cost-conscious applications
   ✅ Straightforward lookups
    """)
    print(f"{'='*80}\n")
