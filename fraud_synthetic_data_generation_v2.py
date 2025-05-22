from faker import Faker
from datetime import datetime, timedelta, date
import pandas as pd
import random
from geopy.distance import geodesic

fake = Faker()

# US ZIP code coordinates
zip_coords = {
    "10001": (40.750742, -73.99653),  # NYC
    "94105": (37.7898, -122.3942),    # SF
    "60601": (41.8853, -87.6229),     # Chicago
    "30301": (33.7489, -84.3881),     # Atlanta
    "75201": (32.7831, -96.8067),     # Dallas
}
zip_codes = list(zip_coords.keys())

# Product options with categories
products = [
    ("iPhone 14 Pro", "electronics"), ("Samsung Galaxy S24", "electronics"),
    ("MacBook Pro", "electronics"), ("Sony PlayStation 5", "electronics"),
    ("Xbox Series X", "electronics"), ("AirPods Max", "electronics"),
    ("Apple Watch Ultra", "electronics"), ("DJI Drone", "electronics"),
    ("Steam Gift Card $100", "digital_goods"), ("Amazon Gift Card $500", "digital_goods"),
    ("Google Play Credit $200", "digital_goods"), ("Crypto Wallet Top-Up", "digital_goods"),
    ("VPN Subscription", "digital_goods"), ("Gaming Credits Purchase", "digital_goods"),
    ("Flight Ticket to Dubai", "travel"), ("Hotel Booking - Las Vegas", "travel"),
    ("Car Rental - BMW 5 Series", "travel"), ("Luxury Cruise Package", "travel"),
    ("Concert Tickets", "travel"), ("Theme Park Annual Pass", "travel"),
    ("Gucci Handbag", "luxury"), ("Louis Vuitton Wallet", "luxury"),
    ("Rolex Submariner", "luxury"), ("Ray-Ban Sunglasses", "luxury"),
    ("Yeezy Sneakers", "luxury"), ("Moncler Jacket", "luxury"),
    ("Balenciaga Hoodie", "luxury"), ("Burberry Trench Coat", "luxury"),
    ("Electric Scooter", "home_appliance"), ("Smart Refrigerator", "home_appliance"),
    ("High-End Gaming PC", "electronics"), ("Designer Perfume", "luxury"),
    ("Apple Vision Pro", "electronics"), ("Premium Mattress", "home_appliance")
]

def generate_fraud_record():
    transaction_zip = random.choice(zip_codes)
    previous_zip = random.choice(zip_codes)
    coords1 = zip_coords[transaction_zip]
    coords2 = zip_coords[previous_zip]
    distance_km = geodesic(coords1, coords2).km
    time_diff_hours = random.uniform(0.5, 3.0)
    is_impossible_travel = distance_km / time_diff_hours > 500

    account_creation_date = fake.date_between(start_date='-3y', end_date='-30d')
    email_change_date = fake.date_between(start_date=account_creation_date, end_date='today')

    average_amount = round(random.uniform(50, 500), 2)
    txn_amount = round(random.uniform(average_amount * 0.5, average_amount * 4), 2)
    credit_limit = random.choice([1000, 2000, 3000, 5000])
    is_spike = txn_amount > average_amount * 2.5
    just_below_limit = (credit_limit - txn_amount) < 5

    item, category = random.choice(products)

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
        "type_of_transaction": random.choice(["online", "POS"]),
        "transaction_amount": txn_amount,
        "average_transaction_amount": average_amount,
        "credit_limit": credit_limit,
        "date_of_last_balance_transfer": fake.date_between(start_date='-180d', end_date='today'),
        "transaction_zip_code": transaction_zip,
        "previous_transaction_zip_code": previous_zip,
        "distance_between_locations_km": round(distance_km, 2),
        "is_impossible_travel": is_impossible_travel,
        "device_used": random.choice(["mobile", "desktop", "tablet"]),
        "is_device_anomaly": random.choice([True, False]),
        "is_geo_anomaly": random.choice([True, False]),
        "date_of_email_change": email_change_date,
        "date_of_phone_change": fake.date_between(start_date=account_creation_date, end_date='today'),
        "date_of_address_change": fake.date_between(start_date=account_creation_date, end_date='today'),
        "date_of_password_change": fake.date_between(start_date=account_creation_date, end_date='today'),
        "date_of_account_creation": account_creation_date,
        "date_of_account_locked": fake.date_between(start_date=account_creation_date, end_date='today'),
        "date_of_credit_limit_changed": fake.date_between(start_date=account_creation_date, end_date='today'),
        "account_age_days": (date.today() - account_creation_date).days,
        "date_of_last_otp_email_challenge": fake.date_between(start_date='-5d', end_date='today'),
        "date_of_last_otp_sms_challenge": fake.date_between(start_date='-5d', end_date='today'),
        "date_of_last_cvv_challenge": fake.date_between(start_date='-5d', end_date='today'),
        "date_of_last_ssn_cvv_challenge": fake.date_between(start_date='-5d', end_date='today'),
        "date_of_last_verid_challenge": fake.date_between(start_date='-5d', end_date='today'),
        "number_of_email_otp_challenges_failed_last_7d": random.randint(0, 5),
        "number_of_sms_otp_challenges_failed_last_7d": random.randint(0, 5),
        "number_of_transactions_last_24h": random.randint(1, 20),
        "number_of_failed_logins_last_24h": random.randint(0, 10),
        "is_transaction_spike": is_spike,
        "is_transaction_just_below_limit": just_below_limit,
        "is_behavioral_anomaly": random.choice([True, False]),
        "is_external_reward_redemption": random.choice([True, False]),
        "is_balance_transfer": random.choice([True, False]),
        "is_fraud": True
    }

# Generate dataset
records = [generate_fraud_record() for _ in range(500)]
df = pd.DataFrame(records)
df.to_csv("fraudulent_credit_card_transactions.csv", index=False)

print("✅ Fraud dataset generated: fraudulent_credit_card_transactions.csv")
