"""
Comprehensive Adversarial Query Evaluation for RAG vs Pure LLM

This script tests both systems with adversarial queries and provides
evidence-based comparison metrics.
"""

import json
import os
import sys
import re
from dotenv import load_dotenv

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    print("⚠️ OpenAI not installed. Install with: pip install openai")
    HAS_OPENAI = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

# Try to import retrieval modules, but don't fail if they're not available
try:
    from retrieval_pipeline.embeddings import embed_texts, embed_query
    from retrieval_pipeline.storage import build_index
    from retrieval_pipeline.retrieval import search
    HAS_PIPELINE = True
except ImportError:
    HAS_PIPELINE = False
    print("⚠️ Retrieval pipeline modules not available. Will test Pure LLM only.")

# Load environment
load_dotenv()

if HAS_OPENAI:
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )
else:
    client = None

# ============================================================================
# PART 1: ADVERSARIAL QUERIES DESIGN
# ============================================================================

ADVERSARIAL_QUERIES = {
    "ambiguous": {
        "query": "What about it?",
        "description": "Extremely ambiguous pronoun reference without context",
        "expected_issues": ["Hallucination likely - LLM will fabricate context"]
    },
    "incomplete": {
        "query": "Is the impact positive?",
        "description": "Incomplete query missing subject - could refer to anything",
        "expected_issues": ["RAG may return irrelevant docs; LLM will guess subject"]
    },
    "misleading": {
        "query": "How did the AI boom affect unemployment rates?",
        "description": "Assumes knowledge not in corpus about AI and unemployment",
        "expected_issues": ["Both systems may hallucinate specifics"]
    },
    "out_of_domain": {
        "query": "What are the best practices for Kubernetes deployment in production?",
        "description": "Query completely outside document domain",
        "expected_issues": ["RAG should return low-confidence results; LLM will hallucinate"]
    }
}

# Ground truth for evaluation
GROUND_TRUTH = {
    "ambiguous": {
        "correct_answer": "Cannot determine - insufficient context",
        "hallucination_threshold": 0.7
    },
    "incomplete": {
        "correct_answer": "Cannot determine - subject unclear",
        "hallucination_threshold": 0.7
    },
    "misleading": {
        "correct_answer": "Information not available in provided documents",
        "hallucination_threshold": 0.8
    },
    "out_of_domain": {
        "correct_answer": "This topic is not covered in the documents",
        "hallucination_threshold": 0.9
    }
}

# ============================================================================
# PART 2: EVALUATION FUNCTIONS
# ============================================================================

def evaluate_accuracy(response, ground_truth, query_type):
    """
    Evaluate accuracy of response against ground truth.
    Returns: accuracy_score (0.0 - 1.0)
    """
    response_lower = response.lower()
    
    # Exact match or semantic equivalence
    if "not available" in response_lower or "cannot determine" in response_lower or "unclear" in response_lower:
        if query_type in ["ambiguous", "incomplete", "misleading", "out_of_domain"]:
            return 1.0  # Correct - acknowledged limitation
    
    # Partial credit for acknowledging insufficient data
    if any(phrase in response_lower for phrase in ["insufficient", "limited information", "not covered", "not found"]):
        return 0.6  # Partial credit
    
    return 0.0  # Response treats uncertain as certain

def detect_hallucination(response, query_type, context=""):
    """
    Detect signs of hallucination in response.
    Returns: hallucination_score (0.0 - 1.0), confidence (0.0 - 1.0)
    """
    hallucination_indicators = [
        ("specific numbers without context", 0.8),
        ("concrete statistics", 0.7),
        ("definitive statements for ambiguous queries", 0.9),
        ("invented facts", 0.9)
    ]
    
    response_lower = response.lower()
    hallucination_score = 0.0
    
    # Check for fabricated specificity
    import re
    has_numbers = bool(re.search(r'\d+%|\d+\.\d+|year \d{4}', response))
    has_dates = bool(re.search(r'\d{4}|\b(January|February|March|April|May|June|July|August|September|October|November|December)\b', response))
    has_specific_names = bool(re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', response))
    
    specificity_score = 0
    if has_numbers:
        specificity_score += 0.3
    if has_dates:
        specificity_score += 0.3
    if has_specific_names:
        specificity_score += 0.2
    
    # For adversarial queries, high specificity = likely hallucination
    if query_type in ["ambiguous", "incomplete", "misleading", "out_of_domain"]:
        hallucination_score = min(specificity_score, 0.95)
    
    confidence = min(abs(specificity_score - 0.5) * 2, 1.0)  # Higher when response is very specific or very vague
    
    return hallucination_score, confidence

def evaluate_consistency(response_set):
    """
    Evaluate consistency across multiple responses to same query.
    response_set: list of responses from same query
    Returns: consistency_score (0.0 - 1.0)
    """
    if len(response_set) <= 1:
        return 1.0
    
    # Simple metric: calculate semantic similarity between responses
    # For now, use word overlap as proxy
    words_per_response = [set(r.lower().split()) for r in response_set]
    
    if not all(words_per_response):
        return 0.0
    
    # Jaccard similarity between first response and others
    similarities = []
    first_words = words_per_response[0]
    
    for other_words in words_per_response[1:]:
        intersection = len(first_words & other_words)
        union = len(first_words | other_words)
        if union > 0:
            similarities.append(intersection / union)
    
    return sum(similarities) / len(similarities) if similarities else 1.0

# ============================================================================
# PART 3: RAG SYSTEM EVALUATION
# ============================================================================

def evaluate_rag_system(query, texts):
    """
    Evaluate RAG system response.
    Returns: {
        'query': query,
        'retrieved_texts': [(text, score), ...],
        'response': llm_response,
        'accuracy': score,
        'hallucination_score': score,
        'retrieval_quality': score
    }
    """
    if not HAS_PIPELINE:
        return None
    
    if not HAS_NUMPY:
        return None
    
    try:
        # Stage 1: Embedding and retrieval
        embeddings = embed_texts(texts)
        index = build_index(embeddings)
        query_emb = embed_query(query)
        
        results = search(index, query_emb, texts, k=3)
        retrieved_texts = [text for text, score in results]
        retrieval_scores = [score for text, score in results]
        
        # Retrieval quality: average score (lower is better in similarity space)
        retrieval_quality = 1.0 - min(np.mean(retrieval_scores), 1.0)
        
        # Stage 2: Generate context
        context = "\n".join(retrieved_texts)
        
        # Stage 3: Create RAG prompt
        prompt = f"""Answer the following question using ONLY the provided context. 
If the answer cannot be found in the context, explicitly state: "This information is not available in the provided documents."

Context:
{context}

Question: {query}

Answer:"""
        
        # Stage 4: Get LLM response
        if not client:
            return None
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        llm_response = response.choices[0].message.content
        
        # Stage 5: Evaluate
        query_type = "out_of_domain"  # Will be overridden by query matching
        for q_type, q_dict in ADVERSARIAL_QUERIES.items():
            if q_dict["query"].lower() == query.lower():
                query_type = q_type
                break
        
        accuracy = evaluate_accuracy(llm_response, GROUND_TRUTH[query_type], query_type)
        hallucination_score, hallucination_conf = detect_hallucination(llm_response, query_type, context)
        
        return {
            'system': 'RAG',
            'query': query,
            'retrieved_texts': results,
            'response': llm_response,
            'accuracy': accuracy,
            'hallucination_score': hallucination_score,
            'hallucination_confidence': hallucination_conf,
            'retrieval_quality': retrieval_quality,
            'context_length': len(context)
        }
    except Exception as e:
        print(f"  ⚠️ RAG evaluation failed: {e}")
        return None

# ============================================================================
# PART 4: PURE LLM EVALUATION
# ============================================================================

def evaluate_pure_llm(query):
    """
    Evaluate pure LLM response (no retrieval).
    Returns: {
        'query': query,
        'response': llm_response,
        'accuracy': score,
        'hallucination_score': score
    }
    """
    if not client:
        return None
    
    try:
        prompt = f"""Answer the following question to the best of your knowledge. 
If you're not certain about the answer, please state that clearly.

Question: {query}

Answer:"""
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        llm_response = response.choices[0].message.content
        
        # Evaluate
        query_type = "out_of_domain"
        for q_type, q_dict in ADVERSARIAL_QUERIES.items():
            if q_dict["query"].lower() == query.lower():
                query_type = q_type
                break
        
        accuracy = evaluate_accuracy(llm_response, GROUND_TRUTH[query_type], query_type)
        hallucination_score, hallucination_conf = detect_hallucination(llm_response, query_type)
        
        return {
            'system': 'Pure LLM',
            'query': query,
            'response': llm_response,
            'accuracy': accuracy,
            'hallucination_score': hallucination_score,
            'hallucination_confidence': hallucination_conf
        }
    except Exception as e:
        print(f"  ⚠️ Pure LLM evaluation failed: {e}")
        return None

# ============================================================================
# PART 5: COMPARISON FRAMEWORK
# ============================================================================

def generate_comparison_report(rag_results, llm_results):
    """
    Generate evidence-based comparison report between RAG and Pure LLM.
    """
    # Calculate averages safely
    if rag_results:
        rag_accuracy = sum(r['accuracy'] for r in rag_results) / len(rag_results) if rag_results else -1
        rag_hallucination = sum(r['hallucination_score'] for r in rag_results) / len(rag_results) if rag_results else -1
    else:
        rag_accuracy = -1
        rag_hallucination = -1
    
    if llm_results:
        llm_accuracy = sum(r['accuracy'] for r in llm_results) / len(llm_results) if llm_results else -1
        llm_hallucination = sum(r['hallucination_score'] for r in llm_results) / len(llm_results) if llm_results else -1
    else:
        llm_accuracy = -1
        llm_hallucination = -1
    
    # Determine winners
    accuracy_winner = "RAG" if rag_accuracy > llm_accuracy else "Pure LLM"
    hallucination_winner = "RAG" if rag_hallucination < llm_hallucination else "Pure LLM"
    
    report = {
        "timestamp": str(json.dumps({})),  # Simplified timestamp
        "methodology": "Controlled comparison on 4 adversarial queries",
        
        "adversarial_queries_tested": list(ADVERSARIAL_QUERIES.keys()),
        
        "accuracy_metrics": {
            "rag": {
                "average": rag_accuracy,
                "details": [{'query': r['query'], 'score': r['accuracy']} for r in rag_results]
            },
            "pure_llm": {
                "average": llm_accuracy,
                "details": [{'query': r['query'], 'score': r['accuracy']} for r in llm_results]
            }
        },
        
        "hallucination_metrics": {
            "rag": {
                "average_score": rag_hallucination,
                "average_confidence": sum(r.get('hallucination_confidence', 0.5) for r in rag_results) / len(rag_results) if rag_results else -1,
                "details": [
                    {
                        'query': r['query'],
                        'hallucination_score': r['hallucination_score'],
                        'confidence': r.get('hallucination_confidence', 0.5)
                    }
                    for r in rag_results
                ]
            },
            "pure_llm": {
                "average_score": llm_hallucination,
                "average_confidence": sum(r.get('hallucination_confidence', 0.5) for r in llm_results) / len(llm_results) if llm_results else -1,
                "details": [
                    {
                        'query': r['query'],
                        'hallucination_score': r['hallucination_score'],
                        'confidence': r.get('hallucination_confidence', 0.5)
                    }
                    for r in llm_results
                ]
            }
        },
        
        "evidence_based_findings": {
            "accuracy_winner": accuracy_winner,
            "hallucination_winner": hallucination_winner,
            
            "rag_advantages": [
                f"Retrieval quality average: {sum(r.get('retrieval_quality', 0) for r in rag_results) / len(rag_results) if rag_results else 0:.3f}" if rag_results else "N/A",
                "Grounded in document context",
                "Can cite sources"
            ],
            
            "pure_llm_advantages": [
                "No retrieval latency",
                "Can answer questions outside document domain",
                "Broader knowledge base"
            ]
        },
        
        "raw_results": {
            "rag": rag_results,
            "pure_llm": llm_results
        }
    }
    
    return report

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Check dependencies
    if not HAS_OPENAI:
        print("❌ OpenAI module required. Install with: pip install openai")
        sys.exit(1)
    
    if not client:
        print("❌ GROQ_API_KEY not found in .env file")
        sys.exit(1)
    
    print("=" * 80)
    print("ADVERSARIAL QUERY EVALUATION: RAG vs Pure LLM")
    print("=" * 80)
    
    # Sample documents for RAG system
    documents = [
        "The election was won by candidate A with 52% of votes.",
        "Candidate B focused on education and healthcare policies.",
        "Economic growth slowed to 2.1% in 2025, down from 3.2% in 2024.",
        "Technology sector investments increased by 15% year-over-year."
    ]
    
    print("\n[PART 1] ADVERSARIAL QUERIES DESIGNED:")
    print("-" * 80)
    for query_type, query_info in ADVERSARIAL_QUERIES.items():
        print(f"\n{query_type.upper()}:")
        print(f"  Query: '{query_info['query']}'")
        print(f"  Description: {query_info['description']}")
        print(f"  Expected Issues: {query_info['expected_issues']}")
    
    print("\n\n[PART 2] RUNNING EVALUATION...")
    print("-" * 80)
    
    rag_results = []
    llm_results = []
    
    for query_type, query_info in ADVERSARIAL_QUERIES.items():
        query = query_info["query"]
        print(f"\nEvaluating: {query_type.upper()}")
        print(f"Query: '{query}'")
        
        # Pure LLM evaluation
        print("  Running Pure LLM...")
        llm_result = evaluate_pure_llm(query)
        if llm_result:
            llm_results.append(llm_result)
            print(f"    Accuracy: {llm_result['accuracy']:.2f}")
            print(f"    Hallucination Score: {llm_result['hallucination_score']:.2f}")
        else:
            print("    ❌ Failed")
        
        # RAG evaluation
        if HAS_PIPELINE:
            print("  Running RAG system...")
            rag_result = evaluate_rag_system(query, documents)
            if rag_result:
                rag_results.append(rag_result)
                print(f"    Accuracy: {rag_result['accuracy']:.2f}")
                print(f"    Hallucination Score: {rag_result['hallucination_score']:.2f}")
            else:
                print("    ⚠️  Skipped (pipeline not fully initialized)")
        else:
            print("  ⚠️  Skipping RAG system (retrieval pipeline not available)")
    
    print("\n\n[PART 3] EVIDENCE-BASED COMPARISON REPORT")
    print("=" * 80)
    
    if llm_results:
        report = generate_comparison_report(rag_results, llm_results)
        
        # Save report
        with open("evaluation_report.json", "w") as f:
            # Convert numpy types for JSON serialization if numpy is available
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print("\nACCURACY COMPARISON:")
        if report['accuracy_metrics']['rag']['average'] >= 0:
            print(f"  RAG Average: {report['accuracy_metrics']['rag']['average']:.3f}")
        if report['accuracy_metrics']['pure_llm']['average'] >= 0:
            print(f"  Pure LLM Average: {report['accuracy_metrics']['pure_llm']['average']:.3f}")
            print(f"  Winner: {report['evidence_based_findings']['accuracy_winner']}")
        
        print("\nHALLUCINATION COMPARISON (lower is better):")
        if report['hallucination_metrics']['rag']['average_score'] >= 0:
            print(f"  RAG Average Score: {report['hallucination_metrics']['rag']['average_score']:.3f}")
        if report['hallucination_metrics']['pure_llm']['average_score'] >= 0:
            print(f"  Pure LLM Average Score: {report['hallucination_metrics']['pure_llm']['average_score']:.3f}")
            print(f"  Winner (lower hallucination): {report['evidence_based_findings']['hallucination_winner']}")
        
        print("\nRAG ADVANTAGES:")
        for advantage in report['evidence_based_findings']['rag_advantages']:
            print(f"  • {advantage}")
        
        print("\nPURE LLM ADVANTAGES:")
        for advantage in report['evidence_based_findings']['pure_llm_advantages']:
            print(f"  • {advantage}")
        
        print("\n✅ Full report saved to: evaluation_report.json")
    else:
        print("❌ No evaluation results generated. Check logs above.")
    
    print("=" * 80)
