# Self-Validating AI Agent for Delivery Feedback Analysis

## Assumptions and Key Decisions

### Core Assumptions
1. **Data Format Assumptions**
   - Feedback is in English text format
   - Rating scale is 1-5 stars
   - Each feedback entry contains delivery-related information
   - Feedback volume is manageable for real-time processing

2. **Validation Requirements**
   - Summary claims should be verifiable against source data
   - A relevance threshold of 10% is sufficient for validation
   - Statistical validation requires confidence intervals
   - Sentiment analysis can be performed using keyword matching

3. **System Operation**
   - Processing time under 2 seconds per validation
   - Memory usage within standard server constraints
   - No real-time update requirements
   - Single language (English) processing

### Key Design Decisions

1. **Modular Architecture**
   - **Decision**: Split into three main components (data generation, validation, reporting)
   - **Why**: Better maintainability and easier testing
   - **Impact**: Clear separation of concerns and simplified debugging

2. **Validation Strategy**
   - **Decision**: Multi-layered validation approach
   - **Components**:
     * Statistical verification with confidence intervals
     * Relevance scoring using keyword density
     * Sentiment consistency checking
   - **Why**: Comprehensive coverage of different validation needs

3. **Random Rating Generation**
   - **Decision**: Generate ratings independently of review text
   - **Why**: Better testing of inconsistency detection
   - **Impact**: More realistic simulation of human behavior

4. **Validation Thresholds**
   - Relevance score threshold: 0.1
   - Confidence interval: 95%
   - Sentiment match required for consistency
   - Minimum word count for validation

5. **Library Choices**
   - spaCy for NLP processing (faster than alternatives)
   - NLTK for specific text processing tasks
   - Built-in Python libraries for statistical calculations

## Project Overview
This project implements a validation system for an AI agent that generates summaries from customer delivery feedback. The system focuses on ensuring accuracy, relevance, and consistency in AI-generated summaries.

## Project Structure

### 1. Sample Data Generation (`sample_data.py`)
```python
generate_sample_feedback()
```
Generates realistic customer feedback data with the following components:

#### Templates:
- Delivery status template: "Delivery arrived [time_status]. Driver was [service_quality]. Package was [condition]."
- Experience template: "My experience with this delivery was [overall]. [specific_detail]."

#### Parameters:
- Time statuses: on time, 2 hours late, 1 day late, early
- Service qualities: very professional, rude, helpful, in a rush
- Package conditions: perfect condition, slightly damaged, completely damaged
- Overall experiences: excellent, terrible, satisfactory, disappointing
- Specific details: various predefined scenarios

#### Output Format:
```json
{
    "id": "FB[1000-9999]",
    "timestamp": "ISO format date",
    "text": "Generated feedback text",
    "rating": 1-5 stars,
    "delivery_metrics": {
        "timeliness": 1-5,
        "driver_courtesy": 1-5,
        "package_condition": 1-5
    }
}
```

### 2. Validation System (`validation.py`)
```python
class SummaryValidator
```
Core validation functionality with multiple validation layers:

#### Initialization
- Loads spaCy English language model
- Sets up NLTK requirements
- Defines delivery-related keywords

#### Key Methods:

1. **Factual Accuracy Validation**
```python
validate_factual_accuracy(summary: str, feedback_data: List[Dict[str, Any]])
```
- Extracts and validates numerical claims
- Calculates confidence intervals
- Identifies unsupported statements
- Performs topic coherence analysis

2. **Relevance Validation**
```python
validate_relevance(summary: str)
```
- Measures delivery-related content density
- Tracks relevant terms
- Calculates relevance scores with threshold of 0.1

3. **Sentiment Consistency**
```python
_check_sentiment_consistency(summary: str, feedback_data: List[Dict[str, Any]])
```
- Analyzes text sentiment
- Compares with average ratings
- Identifies inconsistencies

#### Statistical Analysis
- Confidence interval calculations
- Reliability scoring
- Trend analysis
- Inconsistency detection

### 3. Main Application (`main.py`)
Orchestrates the validation process with these key functions:

1. **Data Generation**
- Generates sample feedback
- Displays formatted feedback with ratings

2. **Summary Validation**
- Processes AI-generated summary
- Runs multiple validation checks
- Generates detailed reports

3. **Results Presentation**
- Factual accuracy analysis
- Relevance metrics
- Sentiment consistency
- Rating distribution
- Inconsistency highlighting

## Installation

1. Dependencies
```bash
pip install -r requirements.txt
```

2. Required Packages
```text
spacy>=3.0.0
nltk>=3.6.0
numpy>=1.19.0
```

3. Language Models
```bash
python -m spacy download en_core_web_sm
```

## Usage

Run the main application:
```bash
python main.py
```

## Output Format

The system provides detailed validation results:

1. **Factual Accuracy**
- Numerical claims analysis
- Statistical verification with confidence intervals
- Unsupported claims identification

2. **Relevance Analysis**
- Overall relevance score
- Delivery terms identified
- Topic coherence measurement

3. **Consistency Analysis**
- Average rating analysis
- Sentiment comparison
- Inconsistency detection

4. **Feedback Analysis**
- Rating distribution
- Detailed inconsistency reports
- Pattern identification

## Limitations and Considerations
- Designed for English language feedback only
- Uses simplified sentiment analysis
- Confidence intervals based on sample size
- Random rating generation in sample data
- Limited to text-based feedback analysis

## Future Enhancements
1. Advanced Analysis
- Deep learning for sentiment analysis
- Pattern recognition
- Temporal trend analysis

2. System Improvements
- Real-time validation
- API integration
- Custom validation rules
- Extended language support

3. Performance Optimization
- Caching mechanisms
- Parallel processing
- Memory optimization

## Technical Notes
- Validation thresholds are configurable
- Statistical confidence level: 95%
- Minimum relevance score: 0.1
- Rating scale: 1-5 stars