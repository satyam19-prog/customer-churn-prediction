import json
import os
import fcntl
from datetime import datetime
import hashlib
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEEDBACK_FILE = os.path.join(BASE_DIR, "backend", "feedback.jsonl")

def save_feedback(customer_profile: dict, plan: dict, rating: int, comment: str = "") -> None:
    try:
        tenure = int(customer_profile.get("tenure_months", 0))
        if tenure <= 12:
            tenure_bucket = "0-12"
        elif tenure <= 24:
            tenure_bucket = "12-24"
        elif tenure <= 48:
            tenure_bucket = "24-48"
        else:
            tenure_bucket = "48+"
        
        contract = customer_profile.get("contract_type", "Unknown")
        risk_tier = plan.get("customer_profile", {}).get("risk_tier", "Unknown")
        
        hash_str = f"{contract}_{tenure_bucket}_{risk_tier}"
        customer_hash = hashlib.md5(hash_str.encode()).hexdigest()
        
        actions = []
        for a in plan.get("recommended_actions", []):
            if "action" in a:
                actions.append(a["action"])
                
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "customer_hash": customer_hash,
            "risk_tier": risk_tier,
            "recommended_actions": actions,
            "rating": rating,
            "comment": comment
        }
        
        with open(FEEDBACK_FILE, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(record) + "\n")
            fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        print(f"Failed to save feedback: {e}")

def get_low_rated_strategies(risk_tier: str) -> list[str]:
    if not os.path.exists(FEEDBACK_FILE):
        return []
        
    rejected_actions = []
    try:
        with open(FEEDBACK_FILE, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if record.get("risk_tier") == risk_tier and record.get("rating") == 0:
                        rejected_actions.extend(record.get("recommended_actions", []))
                except:
                    pass
            fcntl.flock(f, fcntl.LOCK_UN)
            
        counter = Counter(rejected_actions)
        return [action for action, count in counter.items() if count >= 2]
    except Exception as e:
        print(f"Failed to read feedback: {e}")
        return []
