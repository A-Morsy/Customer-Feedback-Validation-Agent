import json
from datetime import datetime, timedelta
import random

def generate_sample_feedback():
    """Generate sample customer feedback data for testing."""
    feedback_templates = [
        {
            "text": "Delivery arrived {time_status}. Driver was {service_quality}. Package was {condition}."
        },
        {
            "text": "My experience with this delivery was {overall}. {specific_detail}."
        }
    ]
    
    time_statuses = ["on time", "2 hours late", "1 day late", "early"]
    service_qualities = ["very professional", "rude", "helpful", "in a rush"]
    conditions = ["in perfect condition", "slightly damaged", "completely damaged"]
    overall_experiences = ["excellent", "terrible", "satisfactory", "disappointing"]
    specific_details = [
        "The driver went above and beyond.",
        "The package was left in the rain.",
        "Communication was great throughout.",
        "No notification was provided about the delivery."
    ]
    
    feedback_data = []
    for _ in range(10):  # Generate 10 sample feedback entries
        template = random.choice(feedback_templates)
        if template == feedback_templates[0]:
            text = template["text"].format(
                time_status=random.choice(time_statuses),
                service_quality=random.choice(service_qualities),
                condition=random.choice(conditions)
            )
        else:
            text = template["text"].format(
                overall=random.choice(overall_experiences),
                specific_detail=random.choice(specific_details)
            )
        
        if template == feedback_templates[0]:
            text = template["text"].format(
                time_status=random.choice(time_statuses),
                service_quality=random.choice(service_qualities),
                condition=random.choice(conditions)
            )
        else:
            text = template["text"].format(
                overall=random.choice(overall_experiences),
                specific_detail=random.choice(specific_details)
            )
            
        # Generate completely random ratings regardless of content
        base_rating = random.randint(1, 5)

        feedback_data.append({
            "id": f"FB{random.randint(1000, 9999)}",
            "timestamp": (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
            "text": text,
            "rating": base_rating,
            "delivery_metrics": {
                "timeliness": random.randint(1, 5),
                "driver_courtesy": random.randint(1, 5),
                "package_condition": random.randint(1, 5)
            }
        })
    
    return feedback_data