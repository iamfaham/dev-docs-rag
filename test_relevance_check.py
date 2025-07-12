#!/usr/bin/env python3
"""
Test script to verify relevance check functionality
"""

import logging
from rag_pipeline import check_relevance, create_context_summary, reset_bm25_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_relevance_check():
    """Test the relevance check with various questions"""

    # React documentation URL
    react_url = "https://react.dev/learn"

    # Reset BM25 data to ensure fresh start
    reset_bm25_data()

    # Test questions
    test_questions = [
        "What are hooks in React?",
        "How do React hooks work?",
        "What is useState hook?",
        "How to use useEffect?",
        "What are React components?",
        "How to create a React component?",
        "What is JSX?",
        "How to manage state in React?",
        "What is the weather today?",  # Should be NO
        "How to cook pasta?",  # Should be NO
    ]

    print("🔍 Testing Relevance Check for React Documentation")
    print("=" * 60)

    # First, let's see the context summary
    print("\n📋 Context Summary:")
    context = create_context_summary(react_url)
    print(context[:1000] + "..." if len(context) > 1000 else context)

    print("\n🧪 Testing Questions:")
    print("-" * 40)

    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")

        try:
            is_relevant = check_relevance(question, react_url)
            result = "✅ RELEVANT" if is_relevant else "❌ NOT RELEVANT"
            print(f"   Result: {result}")

            # Expected results
            expected_relevant = i <= 8  # First 8 should be relevant
            if is_relevant == expected_relevant:
                print("   ✅ CORRECT")
            else:
                print(
                    "   ❌ INCORRECT - Expected:",
                    "RELEVANT" if expected_relevant else "NOT RELEVANT",
                )

        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")

    print("\n" + "=" * 60)
    print("Test completed!")


if __name__ == "__main__":
    test_relevance_check()
