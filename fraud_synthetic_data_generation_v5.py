import pandas as pd
import numpy as np
from faker import Faker
import random
import hashlib
from datetime import datetime, timedelta

# --- Configuration & Seeding ---
fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

# --- Constants & Lookups ---
NUM_CUSTOMERS = 1 # Increased for better credit card specific patterns, can be changed
NUM_ACCOUNTS_PER_CUSTOMER_AVG = 1 # Customers might have more than one credit card

NUM_EVENTS_PER_CUSTOMER_MONTH_AVG = 10 # General customer events
NUM_TRANSACTIONS_PER_ACCOUNT_MONTH_AVG = 10 # Credit cards typically have more transactions
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2023, 2, 1)

# Fraud Injection Rates (percentage of total transactions)
FRAUD_RATE_OVERALL = 0.02 # 2% of all transactions will be fraud, common for CC
FRAUD_TYPE_DISTRIBUTION = {
    'Card Testing': 0.35,       # High for credit cards
    'Account Takeover': 0.25,
    'High-Value Anomaly': 0.20,
    'Synthetic Identity': 0.10,
    'Card Not Present Fraud': 0.05, # New for CC
    'Skimming/Counterfeit': 0.05    # New for CC
}
if sum(FRAUD_TYPE_DISTRIBUTION.values()) != 1.0:
    print("Warning: FRAUD_TYPE_DISTRIBUTION does not sum to 1.0. Adjusting for proportionate split.")
    total = sum(FRAUD_TYPE_DISTRIBUTION.values())
    FRAUD_TYPE_DISTRIBUTION = {k: v / total for k, v in FRAUD_TYPE_DISTRIBUTION.items()}

# --- Helper Functions ---
def hash_info(info):
    """Generates a consistent hash for device/IP info."""
    return hashlib.sha256(str(info).encode()).hexdigest()[:16]

def get_random_datetime_in_range(start, end):
    """Returns a random datetime between start and end."""
    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

# Define various lookups tailored for Credit Cards
CUSTOMER_SEGMENTS = ['Retail', 'Premium', 'Small Business'] # Focused on CC user types
CREDIT_SCORES = ['Excellent', 'Good', 'Fair', 'Poor']
ACCOUNT_TYPES = ['Credit Card'] # Only Credit Card
ACCOUNT_STATUSES = ['Active', 'Closed', 'Delinquent', 'Suspended'] # Credit card specific statuses

ACTIVITY_TYPES = [
    'Login', 'Password Reset', 'Profile Update', 'Credit Limit Increase Request',
    'Statement View', 'Device Registration', 'Payment Made', 'Transaction Dispute'
]
CHALLENGE_TYPES = ['MFA Prompt', 'Captcha', 'Security Question', None]
EVENT_RESULTS = ['Success', 'Failure', 'Cancelled', 'Expired']
DEVICE_TYPES = ['Mobile', 'Desktop', 'Tablet'] # Standard for digital interactions
TRANSACTION_TYPES = ['Purchase', 'Cash Advance', 'Online Transfer', 'Bill Pay', 'Refund', 'Payment'] # Credit card specific
# Expanded MCC Groups crucial for fraud, especially for Card Not Present/Present
MCC_GROUPS = {
    'Retail_General': ['Department Store', 'Grocery Store', 'Pharmacy', 'Online Retailer'],
    'Retail_Specialty': ['Electronics Store', 'Jewelry Store', 'Clothing Store', 'Sporting Goods', 'Online Electronics'],
    'Travel': ['Airline', 'Hotel', 'Car Rental', 'Cruise Line', 'Travel Agency'],
    'Utilities_Services': ['Electric Bill', 'Water Bill', 'Internet Service', 'Phone Bill', 'Restaurant', 'Streaming Service'],
    'Financial_Services': ['Loan Payment', 'Investment Broker', 'Cash Advance'], # Added Cash Advance here
    'High_Risk': ['Online Gambling', 'Crypto Exchange', 'Money Transfer Service', 'Adult Entertainment', 'Dating Service', 'Foreign Exchange Dealer', 'Prepaid Card Issuer'],
    'Cash_Like': ['ATM Withdrawal', 'Money Order Purchase'] # Added ATM here
}
MER_SICS = {sic for group in MCC_GROUPS.values() for sic in group}
ENTRY_MODES = ['Chip', 'Swipe', 'E-commerce', 'Contactless', 'Keyed']
IP_PREFIXES_KNOWN = [f"192.168.{i}." for i in range(5)] + [f"10.0.{i}." for i in range(5)]
IP_PREFIXES_SUSPICIOUS = [f"89.1.{i}." for i in range(3)] + [f"185.2.{i}." for i in range(3)]

# --- Data Generation Functions ---

def generate_customers(n=NUM_CUSTOMERS):
    customers = []
    for i in range(n):
        join_date = fake.date_between(start_date='-10y', end_date=END_DATE - timedelta(days=90))
        name = fake.name()
        
        customers.append({
            "customer_id": f"CUST{i:05}",
            "customer_name": name,
            "customer_address": fake.address().replace('\n', ', '),
            "customer_city": fake.city(),
            "customer_state": fake.state_abbr(),
            "customer_zip": fake.postcode(),
            "customer_country": fake.country_code(),
            "customer_email": fake.email(),
            "customer_phone": fake.phone_number(),
            "customer_dob": fake.date_of_birth(minimum_age=18, maximum_age=80),
            "customer_join_date": join_date,
            "customer_segment": random.choice(CUSTOMER_SEGMENTS),
            "customer_credit_score": random.choice(CREDIT_SCORES),
            "customer_device_fingerprint": hash_info(f"{name}-{fake.uuid4()}"),
            "customer_ip_address": random.choice(IP_PREFIXES_KNOWN) + str(random.randint(1, 254))
        })
    return pd.DataFrame(customers)

def generate_accounts(customers):
    accounts = []
    account_id_counter = 0
    for _, row in customers.iterrows():
        num_accounts = max(1, round(random.gauss(NUM_ACCOUNTS_PER_CUSTOMER_AVG, 0.5)))
        for _ in range(num_accounts):
            acc_type = 'Credit Card' # Always Credit Card
            join_date_dt = pd.to_datetime(row['customer_join_date'])
            open_date = join_date_dt + timedelta(days=random.randint(0, (END_DATE - join_date_dt).days // 2))
            
            credit_limit = round(random.uniform(500, 50000), 2) # Wide range for credit limits
            balance = -1 * round(random.uniform(0, credit_limit * 0.9), 2) # Balance is typically negative (owed) for CC
            
            accounts.append({
                "account_id": f"ACC{account_id_counter:06}",
                "customer_id": row["customer_id"],
                "account_type": acc_type,
                "account_open_date": open_date,
                "account_status": random.choice(ACCOUNT_STATUSES),
                "account_balance": balance, # Represents outstanding balance
                "account_credit_limit": credit_limit,
                "account_currency": "USD"
            })
            account_id_counter += 1
    return pd.DataFrame(accounts)

def generate_events(customers, accounts):
    events = []
    event_id_counter = 0
    customer_account_map = accounts.groupby('customer_id')['account_id'].apply(list).to_dict()

    for _, customer in customers.iterrows():
        join_date_dt = pd.to_datetime(customer['customer_join_date'])
        delta_days = (END_DATE - join_date_dt).days
        
        num_events_total = int(NUM_EVENTS_PER_CUSTOMER_MONTH_AVG * (delta_days / 30))
        
        for _ in range(max(10, num_events_total)):
            ts = get_random_datetime_in_range(join_date_dt, END_DATE)
            
            activity = random.choice(ACTIVITY_TYPES)
            
            result = 'Success'
            challenge = None
            if activity in ['Login', 'Credit Limit Increase Request', 'Transaction Dispute']:
                if random.random() < 0.15:
                    result = 'Failure'
                if result == 'Failure' and random.random() < 0.6:
                    challenge = random.choice([ct for ct in CHALLENGE_TYPES if ct is not None])
                elif result == 'Success' and random.random() < 0.1:
                    challenge = random.choice([ct for ct in CHALLENGE_TYPES if ct is not None])

            event_account_id = None
            if customer['customer_id'] in customer_account_map:
                # Events like Payment Made, Credit Limit Increase Request are account specific
                if activity in ['Payment Made', 'Credit Limit Increase Request', 'Transaction Dispute', 'Statement View']:
                    event_account_id = random.choice(customer_account_map[customer['customer_id']])

            events.append({
                "event_id": f"EVT{event_id_counter:07}",
                "customer_id": customer["customer_id"],
                "account_id": event_account_id, # Can be None for general logins etc.
                "event_timestamp": ts,
                "activity_type": activity,
                "challenge_type": challenge,
                "result": result,
                "event_ip_address": customer['customer_ip_address'] if random.random() < 0.85 else fake.ipv4_public(),
                "event_device_info": customer['customer_device_fingerprint'] if random.random() < 0.85 else hash_info(fake.uuid4()),
                "event_location": f"{fake.city()}, {fake.state_abbr()}"
            })
            event_id_counter += 1
    return pd.DataFrame(events)

def generate_transactions(customers, accounts):
    transactions = []
    transaction_id_counter = 0

    merchants = []
    for i in range(5000):
        mcc_group_key = random.choice(list(MCC_GROUPS.keys()))
        mer_sic = random.choice(MCC_GROUPS[mcc_group_key])
        merchants.append({
            'merchant_id': f'MER{i:05}',
            'merchant_name': fake.company(),
            'merchant_city': fake.city(),
            'merchant_state': fake.state_abbr(),
            'merchant_zip': fake.postcode(),
            'merchant_country': random.choice(['US', 'CA', 'GB', 'DE', 'AU']),
            'mcc_group': mcc_group_key,
            'mer_sic': mer_sic
        })
    merchant_df = pd.DataFrame(merchants)
    
    customer_account_map = accounts.groupby('customer_id')['account_id'].apply(list).to_dict()

    for _, customer in customers.iterrows():
        customer_accounts = customer_account_map.get(customer['customer_id'], [])
        if not customer_accounts: continue

        for account_id in customer_accounts: # Iterate through all customer's credit cards
            account_info = accounts[accounts['account_id'] == account_id].iloc[0]
            join_date_dt = pd.to_datetime(customer['customer_join_date'])
            delta_days = (END_DATE - join_date_dt).days
            
            num_transactions_total = int(NUM_TRANSACTIONS_PER_ACCOUNT_MONTH_AVG * (delta_days / 30))
            
            for _ in range(max(10, num_transactions_total)):
                ts = get_random_datetime_in_range(join_date_dt, END_DATE)
                
                tx_type = random.choice([t for t in TRANSACTION_TYPES if t != 'Payment']) # Payments are events or separate records
                amount = round(random.uniform(5, 1000), 2) # Base amount for CC purchases

                if tx_type == 'Cash Advance':
                    amount = round(random.uniform(100, 2000), 2)
                elif tx_type == 'Online Transfer': # For balance transfers or payments
                    amount = round(random.uniform(50, 3000), 2)
                elif tx_type == 'Bill Pay':
                    amount = round(random.uniform(20, 500), 2)
                elif tx_type == 'Refund':
                    amount = round(random.uniform(10, 500), 2) * -1
                
                card_present = random.choice([True, False])
                entry_mode = 'E-commerce' if not card_present else random.choice(['Chip', 'Swipe', 'Contactless'])
                
                merchant = merchant_df.sample(1).iloc[0]
                
                transactions.append({
                    "transaction_id": f"TXN{transaction_id_counter:07}",
                    "customer_id": customer["customer_id"],
                    "account_id": account_id,
                    "transaction_timestamp": ts,
                    "transaction_amount": amount,
                    "transaction_type": tx_type,
                    "merchant_id": merchant['merchant_id'],
                    "merchant_name": merchant['merchant_name'],
                    "merchant_city": merchant['merchant_city'],
                    "merchant_state": merchant['merchant_state'],
                    "merchant_zip": merchant['merchant_zip'],
                    "merchant_country": merchant['merchant_country'],
                    "mcc_group": merchant['mcc_group'],
                    "mer_sic": merchant['mer_sic'],
                    "currency": "USD",
                    "card_present": card_present,
                    "entry_mode": entry_mode,
                    "device_used": customer['customer_device_fingerprint'],
                    "ip_address": customer['customer_ip_address'],
                    "is_fraud": False,
                    "fraud_type": None
                })
                transaction_id_counter += 1
    return pd.DataFrame(transactions)

# --- Fraud Injection Logic ---

def inject_fraud_patterns(customers_df, accounts_df, events_df, transactions_df, fraud_distribution=FRAUD_TYPE_DISTRIBUTION, overall_fraud_rate=FRAUD_RATE_OVERALL):
    print(f"Injecting fraud patterns... Targeting {overall_fraud_rate*100:.2f}% of transactions.")
    
    num_total_transactions = len(transactions_df)
    target_fraud_count = int(num_total_transactions * overall_fraud_rate)
    
    injected_fraud_transactions = []
    injected_fraud_events = []

    original_transaction_indices_to_remove = set()

    fraud_counts = {k: int(v * target_fraud_count) for k, v in fraud_distribution.items()}
    
    current_total_fraud = sum(fraud_counts.values())
    if current_total_fraud < target_fraud_count:
        largest_cat = max(fraud_counts, key=fraud_counts.get)
        fraud_counts[largest_cat] += (target_fraud_count - current_total_fraud)
    elif current_total_fraud > target_fraud_count:
        largest_cat = max(fraud_counts, key=fraud_counts.get)
        fraud_counts[largest_cat] -= (current_total_fraud - target_fraud_count)
    
    print(f"Fraud distribution targets: {fraud_counts}")

    def get_fraud_timestamp(cust_id):
        cust_txns = transactions_df[transactions_df['customer_id'] == cust_id]
        if not cust_txns.empty:
            return get_random_datetime_in_range(cust_txns['transaction_timestamp'].min(), cust_txns['transaction_timestamp'].max())
        return get_random_datetime_in_range(START_DATE, END_DATE)

    # --- 1. Card Testing ---
    for _ in range(fraud_counts.get('Card Testing', 0)):
        customer = customers_df.sample(1).iloc[0]
        account = accounts_df[accounts_df['customer_id'] == customer['customer_id']].sample(1).iloc[0]
        
        num_test_txns = random.randint(5, 25) # More frequent for CC testing
        start_ts = get_fraud_timestamp(customer['customer_id'])
        
        compromised_ip = random.choice(IP_PREFIXES_SUSPICIOUS) + str(random.randint(1, 254))
        compromised_device = hash_info(fake.uuid4())
        
        for i in range(num_test_txns):
            timestamp = start_ts + timedelta(seconds=random.randint(2, 60 * 3)) # Rapid succession within 3 minutes
            
            merchant_mcc_group = 'High_Risk'
            merchant_sic = random.choice(MCC_GROUPS[merchant_mcc_group])
            
            amount = round(random.uniform(0.10, 2.50), 2) # Very small amounts
            card_present = False
            entry_mode = 'E-commerce' # Typically online for card testing
            
            injected_fraud_transactions.append({
                'transaction_id': f'FTXN_CT{len(injected_fraud_transactions):05}',
                'customer_id': customer['customer_id'],
                'account_id': account['account_id'],
                'transaction_timestamp': timestamp,
                'transaction_amount': amount,
                'transaction_type': 'Purchase',
                'merchant_id': f'MER_CT{len(injected_fraud_transactions):05}',
                'merchant_name': fake.company(),
                'merchant_city': fake.city(),
                'merchant_state': fake.state_abbr(),
                'merchant_zip': fake.postcode(),
                'merchant_country': random.choice(['CN', 'RU', 'BR', 'US', 'GB']), # Mix of domestic/international
                'mcc_group': merchant_mcc_group,
                'mer_sic': merchant_sic,
                'currency': 'USD',
                'card_present': card_present,
                'entry_mode': entry_mode,
                'device_used': compromised_device,
                'ip_address': compromised_ip,
                'is_fraud': True,
                'fraud_type': 'Card Testing'
            })

    # --- 2. Account Takeover (ATO) ---
    for _ in range(fraud_counts.get('Account Takeover', 0)):
        customer = customers_df.sample(1).iloc[0]
        customer_accounts = accounts_df[accounts_df['customer_id'] == customer['customer_id']]
        if customer_accounts.empty: continue
        target_account = customer_accounts.sample(1).iloc[0]

        ato_start_ts = get_fraud_timestamp(customer['customer_id']) - timedelta(days=random.randint(1, 7))
        compromised_ip = random.choice(IP_PREFIXES_SUSPICIOUS) + str(random.randint(1, 254))
        compromised_device = hash_info(fake.uuid4())

        # 1. Failed Login attempts
        for _ in range(random.randint(1, 3)):
            injected_fraud_events.append({
                'event_id': f'FEVT_ATO{len(injected_fraud_events):05}',
                'customer_id': customer['customer_id'],
                'account_id': None,
                'event_timestamp': ato_start_ts + timedelta(minutes=random.randint(1, 10)),
                'activity_type': 'Login',
                'challenge_type': 'None',
                'result': 'Failure',
                'event_ip_address': compromised_ip,
                'event_device_info': compromised_device,
                'event_location': f"{fake.city()}, {fake.state_abbr()}"
            })
        
        # 2. Password Reset / MFA Bypass
        reset_ts = ato_start_ts + timedelta(minutes=random.randint(15, 30))
        injected_fraud_events.append({
            'event_id': f'FEVT_ATO{len(injected_fraud_events):05}',
            'customer_id': customer['customer_id'],
            'account_id': None,
            'event_timestamp': reset_ts,
            'activity_type': 'Password Reset',
            'challenge_type': random.choice(['Email MFA', 'SMS MFA']),
            'result': 'Success', 
            'event_ip_address': compromised_ip,
            'event_device_info': compromised_device,
            'event_location': f"{fake.city()}, {fake.state_abbr()}"
        })

        # 3. Successful Login from new IP/device
        login_ts = reset_ts + timedelta(minutes=random.randint(5, 15))
        injected_fraud_events.append({
            'event_id': f'FEVT_ATO{len(injected_fraud_events):05}',
            'customer_id': customer['customer_id'],
            'account_id': None,
            'event_timestamp': login_ts,
            'activity_type': 'Login',
            'challenge_type': 'None',
            'result': 'Success',
            'event_ip_address': compromised_ip,
            'event_device_info': compromised_device,
            'event_location': f"{fake.city()}, {fake.state_abbr()}"
        })
        
        # 4. Large fraudulent transaction post-ATO
        tx_ts = login_ts + timedelta(minutes=random.randint(2, 20))
        amount = round(random.uniform(1000, 15000), 2) # High value for ATO, often to money mules
        
        merchant_mcc_group = random.choice(['High_Risk', 'Financial_Services'])
        merchant_sic = random.choice(MCC_GROUPS[merchant_mcc_group])

        injected_fraud_transactions.append({
            'transaction_id': f'FTXN_ATO{len(injected_fraud_transactions):05}',
            'customer_id': customer['customer_id'],
            'account_id': target_account['account_id'],
            'transaction_timestamp': tx_ts,
            'transaction_amount': amount,
            'transaction_type': 'Online Transfer', # Common for ATO to external accounts or prepaid cards
            'merchant_id': f'MER_ATO{len(injected_fraud_transactions):05}',
            'merchant_name': fake.company(),
            'merchant_city': fake.city(),
            'merchant_state': fake.state_abbr(),
            'merchant_zip': fake.postcode(),
            'merchant_country': random.choice(['US', 'RU', 'NG', 'BR', 'CN']),
            'mcc_group': merchant_mcc_group,
            'mer_sic': merchant_sic,
            'currency': 'USD',
            'card_present': False, # Online transfer
            'entry_mode': 'E-commerce',
            'device_used': compromised_device,
            'ip_address': compromised_ip,
            'is_fraud': True,
            'fraud_type': 'Account Takeover'
        })
        # Add a second ATO transaction quickly after
        if random.random() < 0.6:
             tx_ts_2 = tx_ts + timedelta(minutes=random.randint(5, 45))
             amount_2 = round(random.uniform(300, 7000), 2)
             injected_fraud_transactions.append({
                'transaction_id': f'FTXN_ATO{len(injected_fraud_transactions):05}',
                'customer_id': customer['customer_id'],
                'account_id': target_account['account_id'],
                'transaction_timestamp': tx_ts_2,
                'transaction_amount': amount_2,
                'transaction_type': random.choice(['Purchase', 'Cash Advance']),
                'merchant_id': f'MER_ATO_2_{len(injected_fraud_transactions):05}',
                'merchant_name': fake.company(),
                'merchant_city': fake.city(),
                'merchant_state': fake.state_abbr(),
                'merchant_zip': fake.postcode(),
                'merchant_country': random.choice(['US', 'RU', 'NG', 'BR', 'CN']),
                'mcc_group': random.choice(['Retail_Specialty', 'High_Risk']),
                'mer_sic': random.choice(list(MER_SICS)),
                'currency': 'USD',
                'card_present': False, # Could be CNP
                'entry_mode': 'E-commerce',
                'device_used': compromised_device,
                'ip_address': compromised_ip,
                'is_fraud': True,
                'fraud_type': 'Account Takeover'
             })


    # --- 3. High-Value Anomaly ---
    for _ in range(fraud_counts.get('High-Value Anomaly', 0)):
        original_tx_candidate = transactions_df.sample(1).iloc[0]
        original_transaction_indices_to_remove.add(original_tx_candidate.name)
        
        customer = customers_df[customers_df['customer_id'] == original_tx_candidate['customer_id']].iloc[0]

        amount = round(original_tx_candidate['transaction_amount'] * random.uniform(5, 30), 2)
        amount = max(amount, 700) # Ensure it's significant

        merchant_mcc_group = random.choice(['High_Risk', 'Retail_Specialty', 'Travel'])
        merchant_sic = random.choice(MCC_GROUPS[merchant_mcc_group])
        
        ip_address = random.choice([customer['customer_ip_address'], random.choice(IP_PREFIXES_SUSPICIOUS) + str(random.randint(1, 254))])
        device_used = random.choice([customer['customer_device_fingerprint'], hash_info(fake.uuid4())])

        injected_fraud_transactions.append({
            'transaction_id': f'FTXN_HVA{len(injected_fraud_transactions):05}',
            'customer_id': original_tx_candidate['customer_id'],
            'account_id': original_tx_candidate['account_id'],
            'transaction_timestamp': original_tx_candidate['transaction_timestamp'],
            'transaction_amount': amount,
            'transaction_type': original_tx_candidate['transaction_type'],
            'merchant_id': original_tx_candidate['merchant_id'],
            'merchant_name': original_tx_candidate['merchant_name'],
            'merchant_city': fake.city() if random.random() < 0.4 else original_tx_candidate['merchant_city'],
            'merchant_state': fake.state_abbr() if random.random() < 0.4 else original_tx_candidate['merchant_state'],
            'merchant_zip': fake.postcode() if random.random() < 0.4 else original_tx_candidate['merchant_zip'],
            'merchant_country': random.choice(['US', 'CA', 'GB', 'DE', 'AU', 'CH']) if random.random() < 0.3 else original_tx_candidate['merchant_country'],
            'mcc_group': merchant_mcc_group,
            'mer_sic': merchant_sic,
            'currency': 'USD',
            'card_present': random.choice([True, False]), # Can be CNP or CP, but value is unusual
            'entry_mode': random.choice(ENTRY_MODES),
            'device_used': device_used,
            'ip_address': ip_address,
            'is_fraud': True,
            'fraud_type': 'High-Value Anomaly'
        })
    
    # --- 4. Synthetic Identity (Credit Card specific focus) ---
    for _ in range(fraud_counts.get('Synthetic Identity', 0)):
        synth_cust_id = f"SCUST{len(customers_df) + len(injected_fraud_transactions):05}"
        synth_customer = {
            "customer_id": synth_cust_id,
            "customer_name": fake.name(),
            "customer_address": fake.address().replace('\n', ', '),
            "customer_city": fake.city(),
            "customer_state": fake.state_abbr(),
            "customer_zip": fake.postcode(),
            "customer_country": fake.country_code(),
            "customer_email": fake.email(),
            "customer_phone": fake.phone_number(),
            "customer_dob": fake.date_of_birth(minimum_age=20, maximum_age=30),
            "customer_join_date": fake.date_between(start_date=END_DATE - timedelta(days=180), end_date=END_DATE - timedelta(days=30)),
            "customer_segment": 'Retail',
            "customer_credit_score": 'Poor', # Often starts with low/no credit
            "customer_device_fingerprint": hash_info(fake.uuid4()),
            "customer_ip_address": random.choice(IP_PREFIXES_SUSPICIOUS) + str(random.randint(1, 254))
        }
        customers_df = pd.concat([customers_df, pd.DataFrame([synth_customer])], ignore_index=True)

        synth_acc_id = f"SACC{len(accounts_df) + len(injected_fraud_transactions):06}"
        synth_account = {
            "account_id": synth_acc_id,
            "customer_id": synth_cust_id,
            "account_type": 'Credit Card',
            "account_open_date": synth_customer['customer_join_date'] + timedelta(days=random.randint(5, 30)),
            "account_status": 'Active',
            "account_balance": -round(random.uniform(500, 2000), 2),
            "account_credit_limit": round(random.uniform(1000, 5000), 2), # Small initial credit limit
            "account_currency": "USD"
        }
        accounts_df = pd.concat([accounts_df, pd.DataFrame([synth_account])], ignore_index=True)

        # Few initial events
        synth_event_start_ts = synth_customer['customer_join_date']
        for i in range(random.randint(1, 3)):
             injected_fraud_events.append({
                'event_id': f'FEVT_SYNTH{len(injected_fraud_events):05}',
                'customer_id': synth_cust_id,
                'account_id': None,
                'event_timestamp': synth_event_start_ts + timedelta(days=random.randint(0, 10)),
                'activity_type': 'Login',
                'challenge_type': None,
                'result': 'Success',
                'event_ip_address': synth_customer['customer_ip_address'],
                'event_device_info': synth_customer['customer_device_fingerprint'],
                'event_location': f"{synth_customer['customer_city']}, {synth_customer['customer_state']}"
            })
        
        # Then burst of activity
        burst_ts = synth_account['account_open_date'] + timedelta(days=random.randint(10, 30))
        num_burst_txns = random.randint(5, 10) # More transactions to quickly max out limit
        for i in range(num_burst_txns):
            timestamp = burst_ts + timedelta(hours=random.randint(0, 48)) # Within a 2-day window
            amount = round(random.uniform(100, 1000), 2) # Significant purchases
            merchant_mcc_group = random.choice(['Retail_Specialty', 'High_Risk', 'Travel'])
            merchant_sic = random.choice(MCC_GROUPS[merchant_mcc_group])
            
            injected_fraud_transactions.append({
                'transaction_id': f'FTXN_SYNTH{len(injected_fraud_transactions):05}',
                'customer_id': synth_cust_id,
                'account_id': synth_acc_id,
                'transaction_timestamp': timestamp,
                'transaction_amount': amount,
                'transaction_type': 'Purchase',
                'merchant_id': f'MER_SYNTH{len(injected_fraud_transactions):05}',
                'merchant_name': fake.company(),
                'merchant_city': fake.city(),
                'merchant_state': fake.state_abbr(),
                'merchant_zip': fake.postcode(),
                'merchant_country': synth_customer['customer_country'],
                'mcc_group': merchant_mcc_group,
                'mer_sic': merchant_sic,
                'currency': "USD",
                'card_present': random.choice([True, False]),
                'entry_mode': random.choice(ENTRY_MODES),
                'device_used': synth_customer['customer_device_fingerprint'],
                'ip_address': synth_customer['customer_ip_address'],
                'is_fraud': True,
                'fraud_type': 'Synthetic Identity'
            })
    
    # --- 5. Card Not Present Fraud (CNP) ---
    for _ in range(fraud_counts.get('Card Not Present Fraud', 0)):
        customer = customers_df.sample(1).iloc[0]
        account = accounts_df[accounts_df['customer_id'] == customer['customer_id']].sample(1).iloc[0]
        
        cnp_ts = get_fraud_timestamp(customer['customer_id'])
        compromised_ip = random.choice(IP_PREFIXES_SUSPICIOUS) + str(random.randint(1, 254))
        compromised_device = hash_info(fake.uuid4())

        amount = round(random.uniform(200, 2000), 2) # Medium to high value online purchase
        
        merchant_mcc_group = random.choice(['Retail_Specialty', 'Travel', 'High_Risk'])
        merchant_sic = random.choice(MCC_GROUPS[merchant_mcc_group])
        
        injected_fraud_transactions.append({
            'transaction_id': f'FTXN_CNP{len(injected_fraud_transactions):05}',
            'customer_id': customer['customer_id'],
            'account_id': account['account_id'],
            'transaction_timestamp': cnp_ts,
            'transaction_amount': amount,
            'transaction_type': 'Purchase',
            'merchant_id': f'MER_CNP{len(injected_fraud_transactions):05}',
            'merchant_name': fake.company(),
            'merchant_city': fake.city(),
            'merchant_state': fake.state_abbr(),
            'merchant_zip': fake.postcode(),
            'merchant_country': random.choice(['CN', 'RU', 'US', 'DE']), # Often international or cross-border
            'mcc_group': merchant_mcc_group,
            'mer_sic': merchant_sic,
            'currency': 'USD',
            'card_present': False, # Key indicator for CNP
            'entry_mode': 'E-commerce',
            'device_used': compromised_device, # Often from a new device/IP
            'ip_address': compromised_ip,
            'is_fraud': True,
            'fraud_type': 'Card Not Present Fraud'
        })
        if random.random() < 0.4: # Sometimes multiple CNP transactions
            cnp_ts_2 = cnp_ts + timedelta(minutes=random.randint(10, 60))
            amount_2 = round(random.uniform(100, 1500), 2)
            injected_fraud_transactions.append({
                'transaction_id': f'FTXN_CNP{len(injected_fraud_transactions):05}',
                'customer_id': customer['customer_id'],
                'account_id': account['account_id'],
                'transaction_timestamp': cnp_ts_2,
                'transaction_amount': amount_2,
                'transaction_type': 'Purchase',
                'merchant_id': f'MER_CNP_2_{len(injected_fraud_transactions):05}',
                'merchant_name': fake.company(),
                'merchant_city': fake.city(),
                'merchant_state': fake.state_abbr(),
                'merchant_zip': fake.postcode(),
                'merchant_country': random.choice(['CN', 'RU', 'US', 'DE']),
                'mcc_group': merchant_mcc_group,
                'mer_sic': merchant_sic,
                'currency': 'USD',
                'card_present': False,
                'entry_mode': 'E-commerce',
                'device_used': compromised_device,
                'ip_address': compromised_ip,
                'is_fraud': True,
                'fraud_type': 'Card Not Present Fraud'
            })


    # --- 6. Skimming/Counterfeit ---
    for _ in range(fraud_counts.get('Skimming/Counterfeit', 0)):
        customer = customers_df.sample(1).iloc[0]
        account = accounts_df[accounts_df['customer_id'] == customer['customer_id']].sample(1).iloc[0]
        
        skimming_ts = get_fraud_timestamp(customer['customer_id'])
        # Often occurs in a different geographic location from customer's normal pattern
        fraud_city = fake.city()
        fraud_state = fake.state_abbr()
        fraud_zip = fake.postcode()
        
        # Transactions are often in-person, but at an unusual location or merchant type
        amount = round(random.uniform(50, 1000), 2)
        
        merchant_mcc_group = random.choice(['Retail_General', 'Travel', 'Cash_Like']) # Places where skimming occurs
        merchant_sic = random.choice(MCC_GROUPS[merchant_mcc_group])
        
        injected_fraud_transactions.append({
            'transaction_id': f'FTXN_SKIM{len(injected_fraud_transactions):05}',
            'customer_id': customer['customer_id'],
            'account_id': account['account_id'],
            'transaction_timestamp': skimming_ts,
            'transaction_amount': amount,
            'transaction_type': 'Purchase',
            'merchant_id': f'MER_SKIM{len(injected_fraud_transactions):05}',
            'merchant_name': fake.company(),
            'merchant_city': fraud_city,
            'merchant_state': fraud_state,
            'merchant_zip': fraud_zip,
            'merchant_country': random.choice(['US', 'MX', 'CA', 'GB']),
            'mcc_group': merchant_mcc_group,
            'mer_sic': merchant_sic,
            'currency': 'USD',
            'card_present': True, # Key indicator for skimming/counterfeit
            'entry_mode': 'Swipe' if random.random() < 0.7 else 'Chip', # Often swipe if card is duplicated
            'device_used': hash_info(f"POS_TERMINAL_{fake.uuid4()}"), # Fraudster's terminal
            'ip_address': fake.ipv4_public(), # Local IP of terminal
            'is_fraud': True,
            'fraud_type': 'Skimming/Counterfeit'
        })
        if random.random() < 0.5: # Follow-up transaction
            skimming_ts_2 = skimming_ts + timedelta(hours=random.randint(1, 24))
            amount_2 = round(random.uniform(30, 500), 2)
            injected_fraud_transactions.append({
                'transaction_id': f'FTXN_SKIM{len(injected_fraud_transactions):05}',
                'customer_id': customer['customer_id'],
                'account_id': account['account_id'],
                'transaction_timestamp': skimming_ts_2,
                'transaction_amount': amount_2,
                'transaction_type': 'Purchase',
                'merchant_id': f'MER_SKIM_2_{len(injected_fraud_transactions):05}',
                'merchant_name': fake.company(),
                'merchant_city': fraud_city, # Same fraud location
                'merchant_state': fraud_state,
                'merchant_zip': fraud_zip,
                'merchant_country': random.choice(['US', 'MX', 'CA', 'GB']),
                'mcc_group': random.choice(['Retail_General', 'Cash_Like']),
                'mer_sic': random.choice(list(MER_SICS)),
                'currency': 'USD',
                'card_present': True,
                'entry_mode': 'Swipe' if random.random() < 0.7 else 'Chip',
                'device_used': hash_info(f"POS_TERMINAL_2_{fake.uuid4()}"),
                'ip_address': fake.ipv4_public(),
                'is_fraud': True,
                'fraud_type': 'Skimming/Counterfeit'
            })


    # Remove original transactions that were "flipped" to fraud scenarios
    if original_transaction_indices_to_remove:
        transactions_df = transactions_df.drop(list(original_transaction_indices_to_remove))
    
    # Concatenate injected fraud transactions/events with the original legitimate ones
    final_transactions_df = pd.concat([transactions_df, pd.DataFrame(injected_fraud_transactions)], ignore_index=True)
    final_events_df = pd.concat([events_df, pd.DataFrame(injected_fraud_events)], ignore_index=True)

    print(f"Total transactions after fraud injection: {len(final_transactions_df)}")
    print(f"Total fraudulent transactions: {final_transactions_df['is_fraud'].sum()}")
    print(f"Fraud type distribution:\n{final_transactions_df['fraud_type'].value_counts(dropna=False)}")
    print(f"Total events after fraud injection: {len(final_events_df)}")
    
    return customers_df, accounts_df, final_events_df, final_transactions_df

# --- Main Execution ---
def generate_all_data():
    print("Generating customer data...")
    customers_df = generate_customers()
    print("Generating account data...")
    accounts_df = generate_accounts(customers_df)
    print("Generating legitimate customer events...")
    events_df = generate_events(customers_df, accounts_df)
    print("Generating legitimate transaction data...")
    transactions_df = generate_transactions(customers_df, accounts_df)

    customers_final_df, accounts_final_df, events_final_df, transactions_final_df = \
        inject_fraud_patterns(customers_df, accounts_df, events_df, transactions_df)
    
    customers_final_df['customer_join_date'] = pd.to_datetime(customers_final_df['customer_join_date'])
    customers_final_df['customer_dob'] = pd.to_datetime(customers_final_df['customer_dob'])
    accounts_final_df['account_open_date'] = pd.to_datetime(accounts_final_df['account_open_date'])
    events_final_df['event_timestamp'] = pd.to_datetime(events_final_df['event_timestamp'])
    transactions_final_df['transaction_timestamp'] = pd.to_datetime(transactions_final_df['transaction_timestamp'])

    return customers_final_df, accounts_final_df, events_final_df, transactions_final_df

if __name__ == "__main__":
    customers_df, accounts_df, events_df, transactions_df = generate_all_data()

    print("\n--- Final Data Head Samples ---")
    print("Customers:\n", customers_df.head())
    print("\nAccounts:\n", accounts_df.head())
    print("\nCustomer Events:\n", events_df.head())
    print("\nTransactions:\n", transactions_df.head())

    print(f"\nTotal Customers: {len(customers_df)}")
    print(f"Total Accounts: {len(accounts_df)}")
    print(f"Total Events: {len(events_df)}")
    print(f"Total Transactions: {len(transactions_df)}")
    print(f"Actual Fraudulent Transactions: {transactions_df['is_fraud'].sum()}")
    print("\nFraud Type Breakdown:")
    print(transactions_df['fraud_type'].value_counts(dropna=False))

    # Optional: Save to CSV
    with pd.ExcelWriter("synthetic_cc_data.xlsx") as writer:
      customers_df.to_excel(writer, sheet_name="Customers", index=False)
      accounts_df.to_excel(writer, sheet_name="Accounts", index=False)
      events_df.to_excel(writer, sheet_name="Events", index=False)
      transactions_df.to_excel(writer, sheet_name="Transactions", index=False)