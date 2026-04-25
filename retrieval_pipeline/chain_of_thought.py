"""
Multi-Step Reasoning with Chain-of-Thought Orchestration

This module implements advanced reasoning that breaks down complex queries
into sub-questions, retrieves context for each, and chains results together
for more accurate and transparent answers.

Novel features:
- Automatic sub-question decomposition
- Step-by-step retrieval and reasoning
- Confidence scoring at each step
- Transparent reasoning chain for interpretability
- Handles multi-hop reasoning (questions requiring multiple document hops)
"""

import json
from typing import List, Dict, Tuple, Any
import numpy as np
from . import retrieval
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# ============================================================================
# STEP 1: QUERY DECOMPOSITION
# ============================================================================

def decompose_query_into_steps(query: str, model: str = "llama-3.3-70b-versatile") -> Dict[str, Any]:
    """
    Break down complex query into sub-questions for step-by-step reasoning.
    
    Args:
        query: The user's complex question
        model: LLM model to use for decomposition
    
    Returns:
        {
            'original_query': str,
            'reasoning_type': 'single_hop' | 'multi_hop' | 'comparative' | 'aggregation',
            'sub_questions': List[str],
            'reasoning_chain': str  # explanation of decomposition
        }
    """
    
    decomposition_prompt = f"""Analyze this query and break it into sub-questions for step-by-step reasoning.

Query: "{query}"

Provide response in JSON format:
{{
    "reasoning_type": "choose from: single_hop (can answer directly), multi_hop (requires connecting multiple documents), comparative (compare entities/concepts), or aggregation (combine multiple pieces of info)",
    "sub_questions": ["sub-question 1", "sub-question 2", ...],
    "reasoning_chain": "Brief explanation of why these sub-questions are needed",
    "complexity_score": (0-1, where 1 is most complex)
}}

Only return valid JSON, no additional text."""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert at breaking down complex questions into reasoning steps. Always return valid JSON."},
                {"role": "user", "content": decomposition_prompt}
            ],
            temperature=0.3  # Low temperature for consistent decomposition
        )
        
        response_text = response.choices[0].message.content
        
        # Parse JSON response
        result = json.loads(response_text)
        result['original_query'] = query
        
        return result
    
    except Exception as e:
        print(f"⚠️ Error in query decomposition: {e}")
        # Fallback to simple decomposition
        return {
            'original_query': query,
            'reasoning_type': 'single_hop',
            'sub_questions': [query],
            'reasoning_chain': 'Direct query (decomposition failed)',
            'complexity_score': 0.3
        }


# ============================================================================
# STEP 2: MULTI-STEP RETRIEVAL
# ============================================================================

def retrieve_for_reasoning_step(
    sub_question: str,
    index: Any,
    texts: List[str],
    step_number: int,
    total_steps: int,
    previous_context: str = ""
) -> Dict[str, Any]:
    """
    Retrieve documents for a specific reasoning step, with context from previous steps.
    
    Args:
        sub_question: The sub-question for this reasoning step
        index: FAISS index
        texts: Text corpus
        step_number: Current step (for tracking)
        total_steps: Total steps in chain
        previous_context: Accumulated context from previous steps
    
    Returns:
        {
            'step': int,
            'sub_question': str,
            'retrieved_docs': [(text, score), ...],
            'context_window': str,  # concatenated context
            'relevance_scores': List[float],
            'confidence': float  # 0-1 confidence in retrieval
        }
    """
    
    # Combine previous context with current question for better retrieval
    enhanced_query = f"{previous_context}\n\nFollow-up: {sub_question}" if previous_context else sub_question
    
    from .embeddings import embed_query
    
    query_emb = embed_query(enhanced_query)
    
    # Adaptive k: increase context for later steps (likely more complex)
    k = min(5, 2 + step_number)  # 2-5 documents based on step
    
    results = retrieval.search(index, query_emb, texts, k=k)
    
    # Extract scores and compute confidence
    retrieved_docs = [(text, score) for text, score in results]
    relevance_scores = [score for _, score in results]
    
    # Confidence: higher scores (lower distance) = higher confidence
    # FAISS returns distances, so lower is better
    avg_relevance = np.mean(relevance_scores) if relevance_scores else 1.0
    confidence = max(0.0, min(1.0, 1.0 - (avg_relevance / 2.0)))  # Normalize to 0-1
    
    context_window = "\n".join([f"[Doc {i+1}] {text}" for i, (text, _) in enumerate(retrieved_docs)])
    
    return {
        'step': step_number,
        'sub_question': sub_question,
        'retrieved_docs': retrieved_docs,
        'context_window': context_window,
        'relevance_scores': [float(s) for s in relevance_scores],
        'confidence': confidence
    }


# ============================================================================
# STEP 3: CHAIN-OF-THOUGHT REASONING
# ============================================================================

def reason_over_retrieval_step(
    step_info: Dict[str, Any],
    model: str = "llama-3.3-70b-versatile"
) -> Dict[str, Any]:
    """
    Use LLM to reason over retrieved documents for a single step.
    
    Args:
        step_info: Output from retrieve_for_reasoning_step()
        model: LLM model to use
    
    Returns:
        {
            'step': int,
            'question': str,
            'reasoning': str,  # LLM's reasoning for this step
            'evidence': str,   # cited evidence
            'conclusion': str, # step conclusion
            'confidence': float
        }
    """
    
    reasoning_prompt = f"""Based on the following documents, answer this question with step-by-step reasoning.

Question: {step_info['sub_question']}

Documents:
{step_info['context_window']}

Provide your response in JSON format:
{{
    "reasoning": "Your step-by-step reasoning process",
    "evidence": "Specific evidence from documents that supports your reasoning",
    "conclusion": "Direct answer to the question",
    "reasoning_confidence": (0-1, based on how well documents support answer)
}}

Return only valid JSON."""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a logical reasoner. Analyze documents carefully and cite evidence. Return valid JSON."},
                {"role": "user", "content": reasoning_prompt}
            ],
            temperature=0.5  # Moderate temperature for balanced reasoning
        )
        
        response_text = response.choices[0].message.content
        result = json.loads(response_text)
        
        return {
            'step': step_info['step'],
            'question': step_info['sub_question'],
            'retrieval_confidence': step_info['confidence'],
            **result  # reasoning, evidence, conclusion, reasoning_confidence
        }
    
    except Exception as e:
        print(f"⚠️ Error in reasoning step {step_info['step']}: {e}")
        return {
            'step': step_info['step'],
            'question': step_info['sub_question'],
            'reasoning': 'Error in reasoning',
            'evidence': '',
            'conclusion': 'Unable to process',
            'reasoning_confidence': 0.0
        }


# ============================================================================
# STEP 4: CHAIN SYNTHESIS
# ============================================================================

def synthesize_chain_of_thought(
    decomposition: Dict[str, Any],
    reasoning_steps: List[Dict[str, Any]],
    model: str = "llama-3.3-70b-versatile"
) -> Dict[str, Any]:
    """
    Synthesize multi-step reasoning into final answer with full transparency.
    
    Args:
        decomposition: Output from decompose_query_into_steps()
        reasoning_steps: List of outputs from reason_over_retrieval_step()
        model: LLM model to use
    
    Returns:
        {
            'original_query': str,
            'reasoning_type': str,
            'steps': List[Dict],  # complete reasoning chain
            'final_answer': str,
            'synthesis_reasoning': str,  # how steps were combined
            'overall_confidence': float,
            'evidence_summary': str,
            'transparency_report': str  # full chain for debugging
        }
    """
    
    # Build synthesis prompt
    steps_summary = "\n\n".join([
        f"Step {s['step']}: {s['question']}\n"
        f"  Reasoning: {s['reasoning']}\n"
        f"  Evidence: {s['evidence']}\n"
        f"  Conclusion: {s['conclusion']}\n"
        f"  Confidence: {s['reasoning_confidence']:.2f}"
        for s in reasoning_steps
    ])
    
    synthesis_prompt = f"""You have worked through multiple reasoning steps to answer this query:

Original Query: {decomposition['original_query']}
Reasoning Type: {decomposition['reasoning_type']}

Step-by-Step Results:
{steps_summary}

Now synthesize these steps into a cohesive final answer. Consider:
1. How do these steps connect?
2. Are there conflicts or consistencies?
3. What is the most reliable conclusion?

Provide response in JSON:
{{
    "final_answer": "Comprehensive answer combining all steps",
    "synthesis_reasoning": "How you combined the step conclusions",
    "overall_confidence": (0-1 confidence in final answer),
    "evidence_summary": "Key evidence supporting final answer",
    "consistency_check": "Any contradictions or agreements between steps"
}}

Return only valid JSON."""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert synthesizer. Combine reasoning steps logically and note any inconsistencies. Return valid JSON."},
                {"role": "user", "content": synthesis_prompt}
            ],
            temperature=0.5
        )
        
        response_text = response.choices[0].message.content
        result = json.loads(response_text)
        
        # Calculate overall confidence
        step_confidences = [s.get('reasoning_confidence', 0.5) for s in reasoning_steps]
        avg_step_confidence = np.mean(step_confidences) if step_confidences else 0.5
        overall_confidence = (result['overall_confidence'] + avg_step_confidence) / 2
        
        return {
            'original_query': decomposition['original_query'],
            'reasoning_type': decomposition['reasoning_type'],
            'complexity_score': decomposition.get('complexity_score', 0.5),
            'steps': reasoning_steps,
            'step_count': len(reasoning_steps),
            'final_answer': result['final_answer'],
            'synthesis_reasoning': result['synthesis_reasoning'],
            'overall_confidence': overall_confidence,
            'evidence_summary': result.get('evidence_summary', ''),
            'consistency_notes': result.get('consistency_check', ''),
            'transparency_report': _generate_transparency_report(decomposition, reasoning_steps, result),
            'metadata': {
                'model': 'llama-3.3-70b-versatile',
                'framework': 'chain-of-thought-orchestration',
                'version': '1.0'
            }
        }
    
    except Exception as e:
        print(f"⚠️ Error in synthesis: {e}")
        # Fallback: concatenate conclusions
        fallback_answer = " ".join([s['conclusion'] for s in reasoning_steps])
        return {
            'original_query': decomposition['original_query'],
            'reasoning_type': decomposition['reasoning_type'],
            'final_answer': fallback_answer,
            'synthesis_reasoning': 'Fallback: concatenated conclusions',
            'overall_confidence': 0.3,
            'steps': reasoning_steps
        }


def _generate_transparency_report(
    decomposition: Dict[str, Any],
    reasoning_steps: List[Dict[str, Any]],
    synthesis: Dict[str, Any]
) -> str:
    """Generate a detailed transparency report showing the full reasoning chain."""
    
    report = f"""
=== CHAIN-OF-THOUGHT REASONING TRANSPARENCY REPORT ===

📋 QUERY ANALYSIS:
  Original Query: {decomposition['original_query']}
  Reasoning Type: {decomposition['reasoning_type']}
  Decomposition Reasoning: {decomposition.get('reasoning_chain', 'N/A')}

🔗 REASONING CHAIN ({len(reasoning_steps)} steps):
"""
    
    for i, step in enumerate(reasoning_steps, 1):
        report += f"""
  Step {i}: {step['question']}
    Retrieval Confidence: {step.get('retrieval_confidence', 'N/A'):.2f}
    Reasoning Confidence: {step.get('reasoning_confidence', 'N/A'):.2f}
    Evidence: {step.get('evidence', '')[:100]}...
    Conclusion: {step.get('conclusion', 'N/A')}
"""
    
    report += f"""
🎯 SYNTHESIS:
  {synthesis.get('synthesis_reasoning', 'N/A')}
  Consistency Notes: {synthesis.get('consistency_check', 'N/A')}

✅ FINAL ANSWER:
  {synthesis.get('final_answer', 'N/A')}
  Overall Confidence: {synthesis.get('overall_confidence', 'N/A')}

=== END REPORT ===
"""
    
    return report


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def chain_of_thought_retrieval_augmented_generation(
    query: str,
    index: Any,
    texts: List[str],
    model: str = "llama-3.3-70b-versatile",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Complete chain-of-thought RAG pipeline.
    
    This is the main entry point that orchestrates:
    1. Query decomposition
    2. Multi-step retrieval
    3. Step-wise reasoning
    4. Chain synthesis
    
    Args:
        query: User question
        index: FAISS index
        texts: Text corpus
        model: LLM model
        verbose: Print step-by-step progress
    
    Returns:
        Complete reasoning output with transparency
    """
    
    if verbose:
        print("\n🔄 CHAIN-OF-THOUGHT RAG PIPELINE")
        print("=" * 60)
    
    # Step 1: Decompose query
    if verbose:
        print("📋 Step 1: Decomposing query...")
    decomposition = decompose_query_into_steps(query, model)
    
    if verbose:
        print(f"   Type: {decomposition['reasoning_type']}")
        print(f"   Sub-questions: {len(decomposition['sub_questions'])}")
    
    # Step 2-3: Retrieve and reason for each sub-question
    reasoning_steps = []
    previous_context = ""
    
    for i, sub_question in enumerate(decomposition['sub_questions'], 1):
        if verbose:
            print(f"\n🔍 Step {i}/{len(decomposition['sub_questions'])}: Retrieving for '{sub_question[:50]}...'")
        
        # Retrieve
        retrieval_info = retrieve_for_reasoning_step(
            sub_question, index, texts, i,
            len(decomposition['sub_questions']),
            previous_context
        )
        
        if verbose:
            print(f"   Confidence: {retrieval_info['confidence']:.2f}")
            print(f"   Retrieved {len(retrieval_info['retrieved_docs'])} documents")
        
        # Reason
        if verbose:
            print(f"   Reasoning...")
        reasoning_result = reason_over_retrieval_step(retrieval_info, model)
        reasoning_steps.append(reasoning_result)
        
        # Accumulate context for next step
        previous_context += f"\nFrom previous step: {reasoning_result['conclusion']}"
        
        if verbose:
            print(f"   Conclusion: {reasoning_result['conclusion'][:60]}...")
    
    # Step 4: Synthesize
    if verbose:
        print(f"\n🎯 Synthesizing reasoning chain...")
    
    final_output = synthesize_chain_of_thought(decomposition, reasoning_steps, model)
    
    if verbose:
        print(f"   Overall Confidence: {final_output['overall_confidence']:.2f}")
        print(f"\n✅ Final Answer:")
        print(f"   {final_output['final_answer'][:100]}...")
        print("=" * 60)
    
    return final_output
