# Smart Credit Scoring System 🎯

A sophisticated credit scoring system that analyzes DeFi wallet transaction data to generate credit scores from 0-1000 without needing any pre-labeled training data. Pretty neat, right?

## What Does This Do? 🤔

This system takes raw blockchain transaction data and figures out who's creditworthy using three different intelligent approaches:

1. **Behavior Pattern Analysis** - Finds the main ways people differ in how they use their wallets
2. **Unusual Activity Detection** - Spots users who behave weirdly (usually means higher risk)
3. **Financial Health Ranking** - Simple ranking based on good vs bad financial indicators

Then it smartly combines all three approaches to create a final credit score that actually makes sense!

## Key Features ✨

- **No Training Data Required** - Works completely unsupervised
- **Multiple Scoring Methods** - Combines 3 different approaches for robust results
- **Smart Weighting** - Automatically figures out which methods work best for your data
- **Real Credit Score Range** - Outputs scores from 0-1000 like real credit bureaus
- **Detailed Analytics** - Shows you exactly how the scoring works

## Quick Start 🚀

### Prerequisites

```bash
pip install pandas numpy scikit-learn scipy
```

### Basic Usage

1. **Prepare your data**: Make sure you have a `user-wallet-transactions.json` file with this structure:
```json
[
    {
        "userWallet": "0x123...",
        "action": "deposit",
        "actionData": {"amount": "1000.0"},
        "timestamp": 1640995200
    }
]
```

2. **Run the scoring**:
```bash
python main.py
```c

3. **Get your results**: Check `wallet_credit_scores.csv` for the final scores!

## How It Actually Works 🧠

### Step 1: Data Processing
The system converts your raw transaction logs into meaningful wallet-level features:
- Transaction counts and volumes
- Repayment ratios and patterns  
- Risk indicators (liquidations, high utilization)
- Activity patterns and diversity

### Step 2: Three-Method Scoring

#### Method 1: Behavior Pattern Analysis
```python
# Uses PCA to find the main ways wallets differ
pca = PCA(n_components=3)
behavior_scores = pca.fit_transform(wallet_features)
```
- Finds natural patterns in how people use their wallets
- No assumptions about what's "good" or "bad"
- Captures ~60% of all variation in the data

#### Method 2: Unusual Activity Detection  
```python
# Isolation Forest spots the weird ones
anomaly_detector = IsolationForest(contamination=0.1)
weirdness_scores = anomaly_detector.decision_function(data)
```
- Normal behavior = good credit, weird behavior = risky
- Automatically identifies outliers without manual rules
- Usually the most important method (80%+ weight)

#### Method 3: Financial Health Ranking
```python
# Simple percentile ranking on key metrics
rank_scores = stats.rankdata(values) / len(values)
```
- Ranks wallets on repayment ratios, liquidation rates, etc.
- Only method that uses domain knowledge about what's good/bad
- Provides stability and interpretability

### Step 3: Smart Combination
```python
# Methods that spread scores more get higher weight
variance = np.var(method_scores)
weight = variance / total_variance
```
- Automatically weights methods based on how well they separate users
- More discriminative methods get more influence
- No hardcoded weights - everything is data-driven!

## Output Explanation 📊

### Credit Score Ranges
- **0-399 (Poor)**: High risk, frequent issues
- **400-549 (Fair)**: Some concerns, manageable risk  
- **550-699 (Good)**: Solid performance, low risk
- **700-799 (Very Good)**: Excellent track record
- **800-899 (Excellent)**: Outstanding creditworthiness
- **900-1000 (Exceptional)**: Elite tier, minimal risk

### Sample Output
```
=== SMART CREDIT SCORING SYSTEM ===
Building credit scores using multiple intelligent approaches...

Analyzing 9 key features: ['n_tx', 'wallet_age_days', 'repay_ratio', ...]

Analyzing user behavior patterns...
  Found patterns explaining [0.221 0.194 0.183]
Looking for unusual activity patterns...
Ranking users by financial health indicators...
Combining all scoring methods intelligently...

How much each method contributes to final score:
  behavior_patterns: 2.4%
  normal_behavior: 84.9%
  financial_health: 12.7%

=== FINAL RESULTS ===
Successfully scored 3,497 wallets
Different scores generated: 529

Credit Score Breakdown:
  Poor (0-399): 81 users (2.3%)
  Fair (400-549): 176 users (5.0%)
  Good (550-699): 549 users (15.7%)
  Very Good (700-799): 881 users (25.2%)
  Excellent (800-899): 923 users (26.4%)
  Exceptional (900-1000): 887 users (25.4%)
```

## Technical Details 🔧

### Supported Transaction Types
- `deposit` - Adding collateral
- `borrow` - Taking loans
- `repay` - Loan repayments  
- `redeemUnderlying` - Withdrawing collateral
- `liquidationCall` - Forced liquidations

### Key Features Generated
| Feature | Description | Impact |
|---------|-------------|---------|
| `repay_ratio` | Total repaid / Total borrowed | Higher = Better |
| `liquidation_rate` | Liquidations / Total transactions | Lower = Better |
| `wallet_age_days` | Days since first transaction | Higher = Better |
| `utilization_ratio` | Borrowed / Deposited | Moderate = Better |
| `action_diversity` | Number of different action types used | Higher = Better |
| `activity_rate` | Transactions per day | Moderate = Better |

### Algorithm Details
- **PCA Components**: Uses top 3 components (typically 50-70% variance)
- **Outlier Contamination**: 10% expected outliers
- **Normalization**: StandardScaler for all numeric features
- **Missing Values**: Forward-filled with 0 (conservative approach)


## File Structure 📁

```
├── main.py                          # Main scoring system
├── user-wallet-transactions.json    # Input transaction data  
├── wallet_credit_scores.csv         # Output scores
└── README.md                        # This file!
```
\