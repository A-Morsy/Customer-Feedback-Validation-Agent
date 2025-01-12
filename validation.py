from typing import List, Dict, Any
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from collections import Counter

class SummaryValidator:
    def __init__(self):
        """Initialize the validator with required NLP models and tools."""
        # Initialize spaCy for text processing
        self.nlp = spacy.load("en_core_web_sm")
        
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        
        # Define keywords related to delivery performance
        self.delivery_keywords = {
            'delivery', 'package', 'shipment', 'driver',
            'time', 'late', 'early', 'condition', 'damaged'
        }

    def validate_factual_accuracy(self, summary: str, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enhanced validation of summary accuracy against feedback data.
        """
        summary_doc = self.nlp(summary)
        
        # 1. Enhanced numerical claims extraction
        numerical_claims = []
        for i, token in enumerate(summary_doc):
            if token.like_num:
                # Get context (previous and next tokens)
                prev_token = token.doc[token.i - 1] if token.i > 0 else None
                next_token = token.doc[token.i + 1] if token.i + 1 < len(token.doc) else None
                
                # Handle percentages and numbers with context
                if next_token and next_token.text == '%':
                    context = self._get_claim_context(token, prev_token)
                    numerical_claims.append({
                        'value': float(token.text),
                        'type': 'percentage',
                        'context': context
                    })
        
        # 2. Calculate actual statistics with confidence intervals
        stats = self._calculate_detailed_statistics(feedback_data)
        
        # 3. Verify each claim against actual data
        claim_verification = []
        for claim in numerical_claims:
            verification = self._verify_numerical_claim(claim, stats)
            claim_verification.append(verification)
        
        # 4. Check for unsupported statements
        unsupported_claims = self._find_unsupported_claims(summary_doc, feedback_data)
        
        # 5. Topic coherence check
        topic_coherence = self._check_topic_coherence(summary_doc, feedback_data)
        
        return {
            "has_numerical_claims": len(numerical_claims) > 0,
            "numerical_claims": numerical_claims,
            "actual_statistics": stats,
            "claim_verification": claim_verification,
            "unsupported_claims": unsupported_claims,
            "topic_coherence": topic_coherence
        }

    def _get_claim_context(self, token, prev_token):
        """Extract context for numerical claims."""
        context_keywords = {
            'delivery': 'delivery_rate',
            'damage': 'damage_rate',
            'satisfaction': 'satisfaction_rate',
            'complaints': 'complaint_rate',
            'positive': 'positive_feedback_rate',
            'negative': 'negative_feedback_rate'
        }
        
        context = 'unknown'
        if prev_token:
            for keyword, claim_type in context_keywords.items():
                if keyword in prev_token.text.lower():
                    context = claim_type
                    break
        
        return context

    def _calculate_detailed_statistics(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate detailed statistics with confidence intervals."""
        total_reviews = len(feedback_data)
        if total_reviews == 0:
            return {}

        # Basic counts
        damage_count = sum(1 for f in feedback_data if 'damaged' in f['text'].lower())
        on_time_count = sum(1 for f in feedback_data if 'on time' in f['text'].lower() or 'early' in f['text'].lower())
        positive_count = sum(1 for f in feedback_data if f['rating'] >= 4)
        
        # Calculate rates with confidence intervals
        def calculate_rate_with_confidence(count):
            rate = (count / total_reviews) * 100
            # Simple confidence interval (can be enhanced with proper statistical methods)
            margin = (1.96 * ((rate * (100 - rate)) / total_reviews) ** 0.5)
            return {
                'rate': round(rate, 1),
                'confidence_interval': (round(max(0, rate - margin), 1), 
                                     round(min(100, rate + margin), 1))
            }

        return {
            "damage_rate": calculate_rate_with_confidence(damage_count),
            "on_time_rate": calculate_rate_with_confidence(on_time_count),
            "satisfaction_rate": calculate_rate_with_confidence(positive_count),
            "total_reviews": total_reviews,
            "reliability_score": min(1.0, total_reviews / 100)  # Simple reliability score
        }

    def _verify_numerical_claim(self, claim, stats):
        """Verify individual numerical claims against statistics."""
        context = claim['context']
        claimed_value = claim['value']
        
        if context in stats:
            actual = stats[context]['rate']
            confidence_interval = stats[context]['confidence_interval']
            is_within_confidence = confidence_interval[0] <= claimed_value <= confidence_interval[1]
            
            return {
                'claim': claimed_value,
                'actual': actual,
                'confidence_interval': confidence_interval,
                'is_accurate': is_within_confidence,
                'deviation': abs(claimed_value - actual),
                'reliability': stats['reliability_score']
            }
        
        return {'claim': claimed_value, 'is_accurate': False, 'reason': 'No matching statistics'}

    def _find_unsupported_claims(self, summary_doc, feedback_data):
        """Identify claims in the summary that aren't supported by feedback."""
        unsupported = []
        
        # Extract key phrases from summary
        summary_phrases = [chunk.text for chunk in summary_doc.noun_chunks]
        
        # Create feedback text index
        feedback_text = ' '.join(f['text'].lower() for f in feedback_data)
        
        # Check each phrase for support
        for phrase in summary_phrases:
            if len(phrase.split()) > 2:  # Only check substantial phrases
                if phrase.lower() not in feedback_text:
                    unsupported.append(phrase)
        
        return unsupported

    def _check_topic_coherence(self, summary_doc, feedback_data):
        """Check if the summary stays on topic with the feedback."""
        # Extract main topics from feedback
        feedback_topics = set()
        for feedback in feedback_data:
            doc = self.nlp(feedback['text'])
            for chunk in doc.noun_chunks:
                if not chunk.root.is_stop:
                    feedback_topics.add(chunk.root.lemma_)

    def validate_relevance(self, summary: str) -> Dict[str, Any]:
        """
        Check if the summary focuses on delivery-related themes.
        
        Args:
            summary: Generated summary text
        
        Returns:
            Dictionary containing relevance validation results
        """
        doc = self.nlp(summary)
        
        # Count delivery-related terms
        delivery_terms = [token.text.lower() for token in doc if token.text.lower() in self.delivery_keywords]
        
        # Calculate relevance score
        words = len([token for token in doc if not token.is_punct])
        relevance_score = len(delivery_terms) / words if words > 0 else 0
        
        return {
            "is_relevant": relevance_score > 0.1,  # Arbitrary threshold
            "relevance_score": relevance_score,
            "delivery_terms_found": delivery_terms  # Added this line
        }

    def _validate_numerical_claims(self, summary_doc, feedback_data: List[Dict[str, Any]]) -> bool:
        """Helper method to validate numerical claims in the summary."""
        # Extract numbers from summary
        summary_numbers = [token.text for token in summary_doc if token.like_num]
        
        # This is a simplified check - in production, would need more sophisticated validation
        return len(summary_numbers) > 0

    def _check_sentiment_consistency(self, summary: str, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check if summary sentiment matches the overall feedback sentiment."""
        # Calculate average rating
        avg_rating = sum(f["rating"] for f in feedback_data) / len(feedback_data)
        
        # Simple sentiment analysis on summary
        positive_words = {'excellent', 'great', 'good', 'positive', 'satisfied'}
        negative_words = {'poor', 'bad', 'negative', 'dissatisfied', 'terrible', 'issues'}
        
        summary_lower = summary.lower()
        positive_count = sum(1 for word in positive_words if word in summary_lower)
        negative_count = sum(1 for word in negative_words if word in summary_lower)
        
        # Determine summary sentiment
        summary_sentiment = "positive" if positive_count > negative_count else "negative"
        feedback_sentiment = "positive" if avg_rating > 3 else "negative"
        
        return {
            "summary_sentiment": summary_sentiment,
            "feedback_sentiment": feedback_sentiment,
            "average_rating": avg_rating,
            "is_consistent": summary_sentiment == feedback_sentiment
        }
