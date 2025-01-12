from sample_data import generate_sample_feedback
from validation import SummaryValidator

def main():
    """Main function to demonstrate the validation system."""
    # Generate multiple sample feedbacks
    feedback_data = generate_sample_feedback()
    
    # Print generated feedback for inspection
    print("\nGenerated Customer Feedback:")
    print("===========================")
    for feedback in feedback_data:
        print(f"\nID: {feedback['id']}")
        print(f"Text: {feedback['text']}")
        print(f"Rating: {'⭐' * feedback['rating']} ({feedback['rating']}/5)")
    
    # Sample AI-generated summary
    sample_summary = """
    Based on recent customer feedback, delivery performance has been mixed. 
    70% of deliveries were on time, with drivers receiving positive feedback for professionalism. 
    However, there were some issues with package handling, as 15% of customers reported damage.
    Communication remains an area for improvement, particularly regarding delivery notifications.
    """
    
    print("\nAI-Generated Summary:")
    print("====================")
    print(sample_summary)
    
    # Initialize and run validator
    validator = SummaryValidator()
    
    # Run validations
    accuracy_results = validator.validate_factual_accuracy(sample_summary, feedback_data)
    relevance_results = validator.validate_relevance(sample_summary)
    
    # Run validations
    validator = SummaryValidator()
    accuracy_results = validator.validate_factual_accuracy(sample_summary, feedback_data)
    relevance_results = validator.validate_relevance(sample_summary)
    sentiment_results = validator._check_sentiment_consistency(sample_summary, feedback_data)
    
    # Print validation results with enhanced details
    print("\nEnhanced Validation Results:")
    print("===========================")
    
    print("\n1. Factual Accuracy Check:")
    print("- Numerical Claims Analysis:")
    for claim in accuracy_results['numerical_claims']:
        print(f"  * Found claim: {claim['value']}% ({claim['context']})")
        
    print("\n- Statistical Verification:")
    for stat_name, stat_data in accuracy_results['actual_statistics'].items():
        if isinstance(stat_data, dict) and 'rate' in stat_data:
            print(f"  * {stat_name}: {stat_data['rate']}% ")
            print(f"    Confidence Interval: {stat_data['confidence_interval']}")
    
    print("\n- Unsupported Claims:")
    for claim in accuracy_results['unsupported_claims']:
        print(f"  * No evidence found for: '{claim}'")
    
    print("\n2. Relevance Analysis:")
    print(f"- Overall relevance score: {relevance_results['relevance_score']:.2f}")
    print("- Delivery terms found:", ', '.join(relevance_results['delivery_terms_found']))
    
    print("\n3. Consistency Analysis:")
    print(f"- Average rating: {sentiment_results['average_rating']:.1f}/5")
    print(f"- Summary tone: {sentiment_results['summary_sentiment']}")
    print(f"- Consistency check: {'Passed' if sentiment_results['is_consistent'] else 'Failed'}")
    
    # Add feedback analysis with enhanced details
    print("\nDetailed Feedback Analysis:")
    print("=========================")
    ratings = [f['rating'] for f in feedback_data]
    print(f"- Rating distribution:")
    for rating in range(1, 6):
        count = ratings.count(rating)
        percentage = (count / len(ratings)) * 100
        print(f"  {rating} stars: {'★' * rating}{'☆' * (5-rating)} ({count} reviews, {percentage:.1f}%)")
    
    print("\n2. Relevance Check:")
    print(f"- Is relevant: {relevance_results['is_relevant']}")
    print(f"- Relevance score: {relevance_results['relevance_score']:.2f}")
    print(f"- Delivery terms found: {relevance_results['delivery_terms_found']}")
    
    print("\n3. Sentiment Consistency Check:")
    print(f"- Summary sentiment: {sentiment_results['summary_sentiment']}")
    print(f"- Average feedback rating: {sentiment_results['average_rating']:.1f}/5")
    print(f"- Is sentiment consistent: {sentiment_results['is_consistent']}")
    
    # Analyze inconsistencies in the feedback
    ratings = [f['rating'] for f in feedback_data]
    avg_rating = sum(ratings) / len(ratings)
    print("\nFeedback Analysis:")
    print("=================")
    print(f"- Average rating: {avg_rating:.1f}/5")
    print("- Potential inconsistencies:")
    for feedback in feedback_data:
        text_lower = feedback['text'].lower()
        # Check for positive words with low ratings
        if any(word in text_lower for word in ['perfect', 'great', 'excellent']) and feedback['rating'] <= 2:
            print(f"  * Low rating ({feedback['rating']}/5) but positive feedback: '{feedback['text']}'")
        # Check for negative words with high ratings
        if any(word in text_lower for word in ['damaged', 'terrible', 'late']) and feedback['rating'] >= 4:
            print(f"  * High rating ({feedback['rating']}/5) but negative feedback: '{feedback['text']}'")

if __name__ == "__main__":
    main()