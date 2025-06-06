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
# Only one customer and one account
NUM_CUSTOMERS = 100
NUM_ACCOUNTS_PER_CUSTOMER_AVG = 1

NUM_EVENTS_TOTAL = 50 # Total events for this customer
NUM_TRANSACTIONS_TOTAL = 100 # Total transactions for this customer's account
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31)

# Fraud Injection Rates (percentage of total transactions)
# We'll aim for a few specific fraud transactions in this small dataset
FRAUD_RATE_OVERALL = 0.05 # 5% of transactions will be fraud
FRAUD_TYPE_DISTRIBUTION = {
    'Card Testing': 0.30,       # 30% of fraud will be card testing
    'Account Takeover': 0.25,   # 25% will be ATO
    'High-Value Anomaly': 0.20, # 20% will be high-value
    'Synthetic Identity': 0.15, # This will be 0 as we only have 1 real customer
    'Loyalty Fraud': 0.10
}
# Adjust for Synthetic Identity not being possible with 1 existing customer
FRAUD_TYPE_DISTRIBUTION['Synthetic Identity'] = 0
total_remaining = sum(v for k, v in FRAUD_TYPE_DISTRIBUTION.items() if k != 'Synthetic Identity')
FRAUD_TYPE_DISTRIBUTION = {k: v / total_remaining if k != 'Synthetic Identity' else 0 for k, v in FRAUD_TYPE_DISTRIBUTION.items()}


# --- Helper Functions ---
def hash_info(info):
    """Generates a consistent hash for device/IP info."""
    return hashlib.sha256(str(info).encode()).hexdigest()[:16]

def get_random_datetime_in_range(start, end):
    """Returns a random datetime between start and end."""
    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

# Define various lookups to make data more realistic
CUSTOMER_SEGMENTS = ['Retail'] # Only one segment for simplicity
CREDIT_SCORES = ['Good'] # Only one credit score for simplicity
ACCOUNT_TYPES = ['Checking'] # Only one account type for simplicity
ACCOUNT_STATUSES = ['Active']
ACTIVITY_TYPES = ['Login', 'Password Reset', 'Profile Update', 'Funds Transfer Attempt', 'Statement View', 'Device Registration', 'Beneficiary Add', 'Address Change']
CHALLENGE_TYPES = ['MFA Prompt', 'Captcha', 'Security Question', None]
EVENT_RESULTS = ['Success', 'Failure', 'Cancelled', 'Expired']
DEVICE_TYPES = ['Mobile', 'Desktop', 'Tablet', 'Smartwatch']
TRANSACTION_TYPES = ['Purchase', 'ATM Withdrawal', 'Online Transfer', 'Bill Pay', 'Deposit', 'Refund', 'Cash Advance']
MCC_GROUPS = {
    'Retail_General': ['Department Store', 'Grocery Store', 'Pharmacy', 'Book Store'],
    'Retail_Specialty': ['Electronics Store', 'Jewelry Store', 'Clothing Store', 'Sporting Goods'],
    'Travel': ['Airline', 'Hotel', 'Car Rental', 'Cruise Line'],
    'Utilities_Services': ['Electric Bill', 'Water Bill', 'Internet Service', 'Phone Bill', 'Restaurant', 'Hair Salon', 'Mechanic'],
    'Financial_Services': ['Bank Fees', 'Loan Payment', 'Investment Broker'],
    'High_Risk': ['Online Gambling', 'Crypto Exchange', 'Money Transfer Service', 'Adult Entertainment', 'Dating Service', 'Foreign Exchange Dealer'],
    'Cash_Like': ['ATM', 'Prepaid Card Load', 'Money Order']
}
MER_SICS = {sic for group in MCC_GROUPS.values() for sic in group}
ENTRY_MODES = ['Chip', 'Swipe', 'E-commerce', 'Contactless', 'Keyed']
IP_PREFIXES_KNOWN = [f"192.168.{i}." for i in range(5)] # Common internal/VPN ranges
IP_PREFIXES_SUSPICIOUS = [f"89.1.{i}." for i in range(3)] # Example suspicious ranges

# --- Data Generation Functions ---

def generate_customers(n=NUM_CUSTOMERS):
    customers = []
    for i in range(n):
        join_date = fake.date_between(start_date='-5y', end_date=END_DATE - timedelta(days=90))
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
        # Force one account per customer
        acc_type = random.choice(ACCOUNT_TYPES)
        join_date_dt = pd.to_datetime(row['customer_join_date'])
        open_date = join_date_dt + timedelta(days=random.randint(0, (END_DATE - join_date_dt).days // 2))

        
        balance = round(random.uniform(1000, 50000), 2) # Checking account balance
        
        accounts.append({
            "account_id": f"ACC{account_id_counter:06}",
            "customer_id": row["customer_id"],
            "account_type": acc_type,
            "account_open_date": open_date,
            "account_status": random.choice(ACCOUNT_STATUSES),
            "account_balance": balance,
            "account_credit_limit": None, # No credit limit for checking
            "account_currency": "USD"
        })
        account_id_counter += 1
    return pd.DataFrame(accounts)

def generate_events(customers, accounts, n_events=NUM_EVENTS_TOTAL):
    events = []
    event_id_counter = 0
    customer_account_map = accounts.groupby('customer_id')['account_id'].apply(list).to_dict()

    for _, customer in customers.iterrows():
        join_date_dt = pd.to_datetime(customer['customer_join_date'])
        delta_days = (END_DATE - join_date_dt).days

        
        for _ in range(n_events):
            ts = get_random_datetime_in_range(join_date_dt, END_DATE)
            
            activity = random.choice(ACTIVITY_TYPES)
            
            result = 'Success'
            challenge = None
            if activity in ['Login', 'Funds Transfer Attempt', 'Password Reset']:
                if random.random() < 0.05: # Lower chance of failure for normal events
                    result = 'Failure'
                if result == 'Failure' and random.random() < 0.6:
                    challenge = random.choice([ct for ct in CHALLENGE_TYPES if ct is not None])

            event_account_id = None
            if customer['customer_id'] in customer_account_map:
                event_account_id = random.choice(customer_account_map[customer['customer_id']])

            events.append({
                "event_id": f"EVT{event_id_counter:07}",
                "customer_id": customer["customer_id"],
                "account_id": event_account_id,
                "event_timestamp": ts,
                "activity_type": activity,
                "challenge_type": challenge,
                "result": result,
                "event_ip_address": customer['customer_ip_address'] if random.random() < 0.9 else fake.ipv4_public(), # Mostly from customer's usual IP
                "event_device_info": customer['customer_device_fingerprint'] if random.random() < 0.9 else hash_info(fake.uuid4()), # Mostly from customer's usual device
                "event_location": f"{fake.city()}, {fake.state_abbr()}"
            })
            event_id_counter += 1
    return pd.DataFrame(events)

def generate_transactions(customers, accounts, n_txns=NUM_TRANSACTIONS_TOTAL):
    transactions = []
    transaction_id_counter = 0

    merchants = []
    for i in range(100): # Fewer merchants for a single customer
        mcc_group = random.choice(list(MCC_GROUPS.keys()))
        mer_sic = random.choice(MCC_GROUPS[mcc_group])
        merchants.append({
            'merchant_id': f'MER{i:05}',
            'merchant_name': fake.company(),
            'merchant_city': fake.city(),
            'merchant_state': fake.state_abbr(),
            'merchant_zip': fake.postcode(),
            'merchant_country': random.choice(['US', 'CA']),
            'mcc_group': mcc_group,
            'mer_sic': mer_sic
        })
    merchant_df = pd.DataFrame(merchants)
    
    customer_account_map = accounts.groupby('customer_id')['account_id'].apply(list).to_dict()

    for _, customer in customers.iterrows():
        customer_accounts = customer_account_map.get(customer['customer_id'], [])
        if not customer_accounts: continue

        account_id = random.choice(customer_accounts) # Pick the single account
        account_info = accounts[accounts['account_id'] == account_id].iloc[0]
        join_date_dt = pd.to_datetime(customer['customer_join_date'])
        
        for _ in range(n_txns):
            ts = get_random_datetime_in_range(join_date_dt, END_DATE)
            
            tx_type = random.choice(TRANSACTION_TYPES)
            amount = round(random.uniform(5, 500), 2) # Base amount for normal transactions for a checking account

            if tx_type == 'ATM Withdrawal':
                amount = round(random.uniform(20, 200), 2)
            elif tx_type == 'Online Transfer':
                amount = round(random.uniform(50, 1000), 2)
            elif tx_type == 'Bill Pay':
                amount = round(random.uniform(20, 500), 2)
            elif tx_type == 'Deposit':
                amount = round(random.uniform(50, 2000), 2)
            elif tx_type == 'Refund':
                amount = round(random.uniform(10, 200), 2) * -1

            card_present = random.choice([True, False])
            entry_mode = 'E-commerce' if not card_present else random.choice(['Chip', 'Swipe', 'Contactless'])
            if tx_type in ['ATM Withdrawal', 'Deposit']:
                card_present = True
                entry_mode = 'Chip'

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
    
    # Adjust for rounding errors
    current_total_fraud = sum(fraud_counts.values())
    if current_total_fraud < target_fraud_count:
        largest_cat = max(fraud_counts, key=fraud_counts.get)
        fraud_counts[largest_cat] += (target_fraud_count - current_total_fraud)
    elif current_total_fraud > target_fraud_count:
        largest_cat = max(fraud_counts, key=fraud_counts.get)
        fraud_counts[largest_cat] -= (current_total_fraud - target_fraud_count)
    
    # Ensure no synthetic identity fraud for one customer scenario
    fraud_counts['Synthetic Identity'] = 0

    print(f"Fraud distribution targets for one customer: {fraud_counts}")

    # Helper to get a random timestamp around a customer's activity
    def get_fraud_timestamp(cust_id):
        cust_txns = transactions_df[transactions_df['customer_id'] == cust_id]
        if not cust_txns.empty:
            return get_random_datetime_in_range(cust_txns['transaction_timestamp'].min(), cust_txns['transaction_timestamp'].max())
        return get_random_datetime_in_range(START_DATE, END_DATE)

    customer = customers_df.iloc[0]
    account = accounts_df.iloc[0]

    # --- 1. Card Testing ---
    for _ in range(fraud_counts.get('Card Testing', 0)):
        num_test_txns = random.randint(3, 7) # Fewer test transactions for a small dataset
        start_ts = get_fraud_timestamp(customer['customer_id'])
        
        compromised_ip = random.choice(IP_PREFIXES_SUSPICIOUS) + str(random.randint(1, 254))
        compromised_device = hash_info(fake.uuid4())
        
        for i in range(num_test_txns):
            timestamp = start_ts + timedelta(seconds=random.randint(5, 60 * 3)) # All within 3 minutes
            merchant_mcc_group = 'High_Risk'
            merchant_sic = random.choice(MCC_GROUPS[merchant_mcc_group])
            amount = round(random.uniform(0.10, 3.00), 2)
            card_present = False
            entry_mode = 'E-commerce'
            
            injected_fraud_transactions.append({
                'transaction_id': f'FTXN_CT{len(injected_fraud_transactions):03}',
                'customer_id': customer['customer_id'],
                'account_id': account['account_id'],
                'transaction_timestamp': timestamp,
                'transaction_amount': amount,
                'transaction_type': 'Purchase',
                'merchant_id': f'MER_CT{len(injected_fraud_transactions):03}',
                'merchant_name': fake.company(),
                'merchant_city': fake.city(),
                'merchant_state': fake.state_abbr(),
                'merchant_zip': fake.postcode(),
                'merchant_country': random.choice(['CN', 'RU', 'US']),
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
        ato_start_ts = get_fraud_timestamp(customer['customer_id']) - timedelta(days=random.randint(1, 3))
        compromised_ip = random.choice(IP_PREFIXES_SUSPICIOUS) + str(random.randint(1, 254))
        compromised_device = hash_info(fake.uuid4())

        # Failed Login attempts
        injected_fraud_events.append({
            'event_id': f'FEVT_ATO{len(injected_fraud_events):03}',
            'customer_id': customer['customer_id'],
            'account_id': None,
            'event_timestamp': ato_start_ts + timedelta(minutes=random.randint(1, 5)),
            'activity_type': 'Login',
            'challenge_type': 'None',
            'result': 'Failure',
            'event_ip_address': compromised_ip,
            'event_device_info': compromised_device,
            'event_location': f"{fake.city()}, {fake.state_abbr()}"
        })
        
        # Password Reset
        reset_ts = ato_start_ts + timedelta(minutes=random.randint(10, 20))
        injected_fraud_events.append({
            'event_id': f'FEVT_ATO{len(injected_fraud_events):03}',
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

        # Successful Login from new IP/device
        login_ts = reset_ts + timedelta(minutes=random.randint(2, 10))
        injected_fraud_events.append({
            'event_id': f'FEVT_ATO{len(injected_fraud_events):03}',
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
        
        # Large fraudulent transaction post-ATO
        tx_ts = login_ts + timedelta(minutes=random.randint(1, 10))
        amount = round(random.uniform(1000, 5000), 2)
        
        merchant_mcc_group = random.choice(['High_Risk', 'Financial_Services'])
        merchant_sic = random.choice(MCC_GROUPS[merchant_mcc_group])

        injected_fraud_transactions.append({
            'transaction_id': f'FTXN_ATO{len(injected_fraud_transactions):03}',
            'customer_id': customer['customer_id'],
            'account_id': account['account_id'],
            'transaction_timestamp': tx_ts,
            'transaction_amount': amount,
            'transaction_type': 'Online Transfer',
            'merchant_id': f'MER_ATO{len(injected_fraud_transactions):03}',
            'merchant_name': fake.company(),
            'merchant_city': fake.city(),
            'merchant_state': fake.state_abbr(),
            'merchant_zip': fake.postcode(),
            'merchant_country': random.choice(['US', 'RU', 'NG']),
            'mcc_group': merchant_mcc_group,
            'mer_sic': merchant_sic,
            'currency': 'USD',
            'card_present': False,
            'entry_mode': 'E-commerce',
            'device_used': compromised_device,
            'ip_address': compromised_ip,
            'is_fraud': True,
            'fraud_type': 'Account Takeover'
        })

    # --- 3. High-Value Anomaly ---
    for _ in range(fraud_counts.get('High-Value Anomaly', 0)):
        # Select an existing legitimate transaction to "flip" and modify
        original_tx_candidate = transactions_df.sample(1).iloc[0]
        original_transaction_indices_to_remove.add(original_tx_candidate.name)
        
        amount = round(original_tx_candidate['transaction_amount'] * random.uniform(10, 30), 2)
        amount = max(amount, 500)

        merchant_mcc_group = random.choice(['High_Risk', 'Retail_Specialty'])
        merchant_sic = random.choice(MCC_GROUPS[merchant_mcc_group])
        
        ip_address = random.choice([customer['customer_ip_address'], random.choice(IP_PREFIXES_SUSPICIOUS) + str(random.randint(1, 254))])
        device_used = random.choice([customer['customer_device_fingerprint'], hash_info(fake.uuid4())])

        injected_fraud_transactions.append({
            'transaction_id': f'FTXN_HVA{len(injected_fraud_transactions):03}',
            'customer_id': original_tx_candidate['customer_id'],
            'account_id': original_tx_candidate['account_id'],
            'transaction_timestamp': original_tx_candidate['transaction_timestamp'],
            'transaction_amount': amount,
            'transaction_type': original_tx_candidate['transaction_type'],
            'merchant_id': original_tx_candidate['merchant_id'],
            'merchant_name': original_tx_candidate['merchant_name'],
            'merchant_city': fake.city() if random.random() < 0.5 else original_tx_candidate['merchant_city'],
            'merchant_state': fake.state_abbr() if random.random() < 0.5 else original_tx_candidate['merchant_state'],
            'merchant_zip': fake.postcode() if random.random() < 0.5 else original_tx_candidate['merchant_zip'],
            'merchant_country': random.choice(['US', 'CA', 'GB']) if random.random() < 0.3 else original_tx_candidate['merchant_country'],
            'mcc_group': merchant_mcc_group,
            'mer_sic': merchant_sic,
            'currency': 'USD',
            'card_present': random.choice([True, False]),
            'entry_mode': random.choice(ENTRY_MODES),
            'device_used': device_used,
            'ip_address': ip_address,
            'is_fraud': True,
            'fraud_type': 'High-Value Anomaly'
        })
    
    # --- 4. Loyalty Fraud (New Pattern) ---
    for _ in range(fraud_counts.get('Loyalty Fraud', 0)):
        loyalty_transfer_ts = get_fraud_timestamp(customer['customer_id'])
        compromised_ip = random.choice(IP_PREFIXES_SUSPICIOUS) + str(random.randint(1, 254))
        compromised_device = hash_info(fake.uuid4())

        injected_fraud_events.append({
            'event_id': f'FEVT_LOYALTY{len(injected_fraud_events):03}',
            'customer_id': customer['customer_id'],
            'account_id': None,
            'event_timestamp': loyalty_transfer_ts - timedelta(minutes=random.randint(1, 5)),
            'activity_type': 'Login',
            'challenge_type': None,
            'result': 'Success',
            'event_ip_address': compromised_ip,
            'event_device_info': compromised_device,
            'event_location': f"{fake.city()}, {fake.state_abbr()}"
        })
        
        amount = round(random.uniform(30, 200), 2)
        injected_fraud_transactions.append({
            'transaction_id': f'FTXN_LOYALTY{len(injected_fraud_transactions):03}',
            'customer_id': customer['customer_id'],
            'account_id': None,
            'transaction_timestamp': loyalty_transfer_ts,
            'transaction_amount': amount,
            'transaction_type': 'Points Redemption',
            'merchant_id': f'MER_LOYALTY{len(injected_fraud_transactions):03}',
            'merchant_name': random.choice(['Loyalty Partner A', 'Travel Rewards Co.']),
            'merchant_city': fake.city(),
            'merchant_state': fake.state_abbr(),
            'merchant_zip': fake.postcode(),
            'merchant_country': random.choice(['US', 'CA']),
            'mcc_group': 'Services',
            'mer_sic': 'Loyalty Program Service',
            'currency': 'Points',
            'card_present': False,
            'entry_mode': 'E-commerce',
            'device_used': compromised_device,
            'ip_address': compromised_ip,
            'is_fraud': True,
            'fraud_type': 'Loyalty Fraud'
        })


    # Remove original transactions that were "flipped" to fraud scenarios
    if original_transaction_indices_to_remove:
        transactions_df = transactions_df.drop(list(original_transaction_indices_to_remove))
    
    # Concatenate injected fraud transactions/events
    final_transactions_df = pd.concat([transactions_df, pd.DataFrame(injected_fraud_transactions)], ignore_index=True)
    final_events_df = pd.concat([events_df, pd.DataFrame(injected_fraud_events)], ignore_index=True)

    print(f"Total transactions after fraud injection: {len(final_transactions_df)}")
    print(f"Total fraudulent transactions: {final_transactions_df['is_fraud'].sum()}")
    print(f"Fraud type distribution:\n{final_transactions_df['fraud_type'].value_counts(dropna=False)}")
    print(f"Total events after fraud injection: {len(final_events_df)}")
    
    return customers_df, accounts_df, final_events_df, final_transactions_df

# --- Main Execution ---
def generate_all_data_for_single_customer():
    print("Generating single customer data...")
    customers_df = generate_customers(NUM_CUSTOMERS)
    print("Generating single account data...")
    accounts_df = generate_accounts(customers_df)
    print("Generating customer events...")
    events_df = generate_events(customers_df, accounts_df, n_events=NUM_EVENTS_TOTAL)
    print("Generating transaction data...")
    transactions_df = generate_transactions(customers_df, accounts_df, n_txns=NUM_TRANSACTIONS_TOTAL)

    # Inject fraud into the generated legitimate data
    customers_final_df, accounts_final_df, events_final_df, transactions_final_df = \
        inject_fraud_patterns(customers_df, accounts_df, events_df, transactions_df)
    
    # Ensure all timestamp columns are datetime objects
    customers_final_df['customer_join_date'] = pd.to_datetime(customers_final_df['customer_join_date'])
    customers_final_df['customer_dob'] = pd.to_datetime(customers_final_df['customer_dob'])
    accounts_final_df['account_open_date'] = pd.to_datetime(accounts_final_df['account_open_date'])
    events_final_df['event_timestamp'] = pd.to_datetime(events_final_df['event_timestamp'])
    transactions_final_df['transaction_timestamp'] = pd.to_datetime(transactions_final_df['transaction_timestamp'])

    return customers_final_df, accounts_final_df, events_final_df, transactions_final_df

if __name__ == "__main__":
    customers_df, accounts_df, events_df, transactions_df = generate_all_data_for_single_customer()

    print("\n--- Final Data Head Samples (Single Customer) ---")
    print("Customers:\n", customers_df) # No head as it's just one row
    print("\nAccounts:\n", accounts_df) # No head as it's just one row
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
    customers_df.to_csv("synthetic_single_customer.csv", index=False)
    accounts_df.to_csv("synthetic_single_account.csv", index=False)
    events_df.to_csv("synthetic_single_customer_events.csv", index=False)
    transactions_df.to_csv("synthetic_single_customer_transactions.csv", index=False)