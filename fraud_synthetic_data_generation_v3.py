from faker import Faker
from datetime import datetime, timedelta, date
import pandas as pd
import random
from geopy.distance import geodesic

# Initialize Faker for generating realistic-looking data
fake = Faker()

# Predefined US ZIP code coordinates for demonstrating impossible travel
zip_coords = {
    "10001": (40.750742, -73.99653),  # New York City
    "94105": (37.7898, -122.3942),    # San Francisco
    "60601": (41.8853, -87.6229),     # Chicago
    "30301": (33.7489, -84.3881),     # Atlanta
    "75201": (32.7831, -96.8067),     # Dallas
    "90210": (34.0901, -118.4065),    # Beverly Hills
    "33101": (25.7738, -80.1937),     # Miami
    "80202": (39.7533, -104.9961)     # Denver
}
zip_codes = list(zip_coords.keys())

# Product options with categories, heavily biased towards high-risk items for fraud
products = [
    ("iPhone 15 Pro", "electronics"), ("Samsung Galaxy S25", "electronics"),
    ("High-End Gaming PC", "electronics"), ("Luxury Smartwatch", "electronics"),
    ("Steam Gift Card $250", "digital_goods"), ("Crypto Wallet Top-Up (High Value)", "digital_goods"),
    ("Premium VPN Subscription", "digital_goods"), ("Online Casino Credits", "digital_goods"),
    ("First Class Flight Ticket - Global", "travel"), ("Luxury Resort Stay - Maldives", "travel"),
    ("Exotic Car Rental (Ferrari)", "travel"), ("Exclusive Concert VIP Pass", "travel"),
    ("Hermes Birkin Bag", "luxury"), ("Diamond Necklace", "luxury"),
    ("Rolex Daytona", "luxury"), ("Designer Sneaker Collection", "luxury"),
    ("Betting Credits", "gambling"), ("NFT Purchase", "crypto") # Added more specific high-risk
]

def generate_fraud_record():
    transaction_zip = random.choice(zip_codes)
    
    # Ensure impossible travel: pick a previous zip code that is guaranteed to be far
    possible_previous_zips_for_impossible_travel = [
        zc for zc in zip_codes if geodesic(zip_coords[transaction_zip], zip_coords[zc]).km > 2500
    ]
    if possible_previous_zips_for_impossible_travel:
        previous_zip = random.choice(possible_previous_zips_for_impossible_travel)
    else: # Fallback if no sufficiently distant zip found (shouldn't happen with current data)
        previous_zip = random.choice([zc for zc in zip_codes if zc != transaction_zip])

    coords1 = zip_coords[transaction_zip]
    coords2 = zip_coords[previous_zip]
    distance_km = geodesic(coords1, coords2).km
    
    # Time difference made extremely small to reinforce impossible travel
    time_diff_hours = random.uniform(0.005, 0.05) # 0.3 to 3 minutes
    is_impossible_travel = True # Always True for fraudulent records

    # Account creation date to suggest a dormant to active transition, but still relatively new for ATO
    account_creation_date = fake.date_between(start_date='-2y', end_date='-6m') 
    is_dormant_to_active_transition = True # Always true for fraud

    # All security-related dates are very recent to indicate suspicious activity (within hours/days)
    today_dt = date.today()
    transaction_date = fake.date_between(start_date=today_dt - timedelta(days=2), end_date='today') # Transaction happened very recently

    # Temporal Anomalies: Force transaction to odd hours or weekends
    transaction_time = None
    is_odd_hour_activity = False
    is_weekend_activity = False

    if random.random() < 0.8: # 80% chance of odd hour
        # Odd hours (e.g., 2 AM - 5 AM local time)
        hour = random.randint(2, 5)
        minute = random.randint(0, 59)
        transaction_time = datetime(transaction_date.year, transaction_date.month, transaction_date.day, hour, minute)
        is_odd_hour_activity = True
    else: # 20% chance of random time, but still potentially weekend
        transaction_time = fake.date_time_between(start_date='-2d', end_date='now') # Recent random time
    
    # Check if it's a weekend for the transaction date
    if transaction_date.weekday() >= 5: # Saturday or Sunday
        is_weekend_activity = True

    # Personal Info Change + Transaction
    # Ensure these dates are very recent, preceding the transaction by a very short time
    email_change_date = fake.date_time_between(start_date=transaction_time - timedelta(hours=10), end_date=transaction_time - timedelta(minutes=1))
    date_of_phone_change = fake.date_time_between(start_date=transaction_time - timedelta(hours=10), end_date=transaction_time - timedelta(minutes=1))
    date_of_address_change = fake.date_time_between(start_date=transaction_time - timedelta(hours=10), end_date=transaction_time - timedelta(minutes=1))
    # Corrected keyword arguments here: start_date and end_date
    date_of_password_change = fake.date_time_between(start_date=transaction_time - timedelta(hours=10), end_date=transaction_time - timedelta(minutes=1))
    is_personal_info_change_followed_by_txn = True # Always true for fraud

    date_of_account_locked = fake.date_time_between(start_date=transaction_time - timedelta(hours=2), end_date=transaction_time) # Very recent lock
    date_of_credit_limit_changed = fake.date_time_between(start_date=transaction_time - timedelta(days=5), end_date=transaction_time - timedelta(hours=1)) # Recent limit change


    average_amount = round(random.uniform(20, 200), 2) # Lower average to show greater 'spike'
    credit_limit = random.choice([1000, 2500, 5000, 10000, 20000])

    # Highly emphasize transaction spikes or just-below-limit
    is_spike = False
    just_below_limit = False
    if random.random() < 0.7: # 70% chance of a very high spike
        txn_amount = round(random.uniform(average_amount * 10, average_amount * 25), 2) # Much, much higher spike
        is_spike = True
    else: # 30% chance of just below limit (very precisely)
        txn_amount = round(credit_limit - random.uniform(0.01, 2.00), 2) # Exactly just below
        just_below_limit = True
    
    # Ensure txn_amount is within a somewhat plausible range even with spikes
    if txn_amount > credit_limit * 2.5 and not just_below_limit: # If an extreme spike, ensure it's still plausible
        txn_amount = round(random.uniform(credit_limit * 1.1, credit_limit * 2.0), 2) # Just over limit, still a spike

    # Change in Average Daily Spend
    change_in_avg_daily_spend_ratio = round(txn_amount / average_amount, 2) # Show the ratio clearly

    item, category = random.choice(products)
    # Further bias towards high-risk categories for fraud
    if random.random() < 0.9: # 90% chance of high-risk category
        item, category = random.choice([p for p in products if p[1] in ["digital_goods", "luxury", "travel", "gambling", "crypto"]])
    
    # Specific flags for product risk
    is_unusual_mcc_category = random.random() < 0.05, # 5% chance of being true
    is_new_unknown_merchant = random.random() < 0.05, # 5% chance of being true

    # Other anomaly indicators
    is_high_transaction_speed = random.random() < 0.05, # 5% chance of being true
    is_unusual_input_patterns = random.random() < 0.05, # 5% chance of being true
    is_device_emulation_detected = random.random() < 0.05, # 5% chance of being true

    # Advanced and emerging risks
    is_synthetic_identity_behavior = random.random() < 0.05, # 5% chance of being true
    is_deepfake_voiceprint_abuse_detected = random.random() < 0.05, # 5% chance of being true
    ml_behavioral_anomaly_score = round(random.uniform(0.9, 0.99), 2) # High ML anomaly score


    return {
        "transaction_id": fake.uuid4(),
        "customer_name": fake.name(),
        "customer_address": fake.address(),
        "customer_ip_address": fake.ipv4_public(),
        "merchant_name": fake.company(),
        "merchant_address": fake.address(),
        "merchant_ip_address": fake.ipv4_public(),
        "item_purchased": item,
        "product_category": category,
        "activity_type": "purchase",
        "type_of_transaction": random.choice(["online", "POS", "mobile_app"]), # Added mobile_app channel
        "transaction_amount": txn_amount,
        "average_transaction_amount": average_amount,
        "credit_limit": credit_limit,
        "date_of_last_balance_transfer": fake.date_between(start_date='-180d', end_date='today'),
        "transaction_zip_code": transaction_zip,
        "previous_transaction_zip_code": previous_zip,
        "distance_between_locations_km": round(distance_km, 2),
        "is_impossible_travel": is_impossible_travel,
        "device_used": random.choice(["mobile", "desktop", "tablet"]),
        "is_device_anomaly": random.random() < 0.05, # 5% chance of being true
        "is_geo_anomaly": random.random() < 0.05, # 5% chance of being true
        "is_new_device_login": random.random() < 0.05, # 5% chance of being true
        "is_new_ip_login": random.random() < 0.05, # 5% chance of being true
        "is_geo_change_login": random.random() < 0.05, # 5% chance of being true
        "is_sim_swap_detected": random.random() < 0.05, # 5% chance of being true
        "date_of_email_change": email_change_date,
        "date_of_phone_change": date_of_phone_change,
        "date_of_address_change": date_of_address_change,
        "date_of_password_change": date_of_password_change, # Corrected
        "date_of_account_creation": account_creation_date,
        "date_of_account_locked": date_of_account_locked,
        "date_of_credit_limit_changed": date_of_credit_limit_changed,
        "account_age_days": (today_dt - account_creation_date).days,
        "date_of_last_otp_email_challenge": fake.date_time_between(start_date=transaction_time - timedelta(hours=1), end_date=transaction_time),
        "date_of_last_otp_sms_challenge": fake.date_time_between(start_date=transaction_time - timedelta(hours=1), end_date=transaction_time),
        "date_of_last_cvv_challenge": fake.date_time_between(start_date=transaction_time - timedelta(hours=1), end_date=transaction_time),
        "date_of_last_ssn_cvv_challenge": fake.date_time_between(start_date=transaction_time - timedelta(hours=1), end_date=transaction_time),
        "date_of_last_verid_challenge": fake.date_time_between(start_date=transaction_time - timedelta(hours=1), end_date=transaction_time),
        "number_of_email_otp_challenges_failed_last_7d": random.randint(10, 50), # Extremely high
        "number_of_sms_otp_challenges_failed_last_7d": random.randint(10, 50),
        "number_of_transactions_last_24h": random.randint(30, 100), # Very high frequency
        "number_of_failed_logins_last_24h": random.randint(15, 40), # Very high number
        "is_transaction_spike": is_spike,
        "is_transaction_just_below_limit": just_below_limit,
        "is_external_reward_redemption": random.random() < 0.05, # 5% chance of being true
        "is_balance_transfer": random.choice([True, False]),
        

        # New fields for more detailed patterns - probabilities adjusted
        "transaction_timestamp": transaction_time.strftime('%Y-%m-%d %H:%M:%S'), # Explicit timestamp
        "is_odd_hour_activity": is_odd_hour_activity,
        "is_weekend_activity": is_weekend_activity,
        "is_microtransactions_flood": random.random() < 0.1, # 10% chance of being true
        "number_of_frequent_reversals_declines_last_7d": random.randint(5, 20), # High count
        "is_cross_channel_inconsistency": random.random() < 0.2, # 20% chance of being true
        "is_unusual_mcc_category": is_unusual_mcc_category,
        "is_new_unknown_merchant": is_new_unknown_merchant,
        "is_spending_category_shift": random.random() < 0.15, # 15% chance of being true
        "change_in_avg_daily_spend_ratio": change_in_avg_daily_spend_ratio, # Ratio of current txn to average
        "is_dormant_to_active_transition": is_dormant_to_active_transition,
        "is_high_transaction_speed": is_high_transaction_speed,
        "is_unusual_input_patterns": is_unusual_input_patterns,
        "is_device_emulation_detected": is_device_emulation_detected,
        "is_personal_info_change_followed_by_txn": is_personal_info_change_followed_by_txn,
        "is_challenge_bypassed_skipped": random.random() < 0.15, # 15% chance of being true
        "is_synthetic_identity_behavior": is_synthetic_identity_behavior,
        "is_deepfake_voiceprint_abuse_detected": is_deepfake_voiceprint_abuse_detected,
        "ml_behavioral_anomaly_score": ml_behavioral_anomaly_score,
        "is_behavioral_anomaly": random.random() < 0.1, # 10% chance of being true
        "is_fraud": True # Always True for this dataset
    }

# Generate ONLY fraudulent records
num_fraudulent = 1000 # You can adjust the number of records as needed

fraudulent_records = [generate_fraud_record() for _ in range(num_fraudulent)]
df_fraudulent = pd.DataFrame(fraudulent_records)

# Save the dataset to a CSV file
output_filename = "final_enhanced_fraudulent_transactions_only.csv"
df_fraudulent.to_csv(output_filename, index=False)

print(f"✅ Final enhanced fraudulent transactions dataset generated: {output_filename}")
print("This dataset contains only fraudulent transactions, with specified patterns applied probabilistically to show varying fraud behaviors.")