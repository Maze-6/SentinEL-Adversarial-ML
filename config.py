# System Configuration
# Lists used for heuristic analysis and allowlisting

# Trusted domains to prevent false positives on popular sites
TRUSTED_DOMAINS = [
    # Tech & Infrastructure
    "google.com", "amazon.com", "microsoft.com", "github.com", "apple.com",
    "linkedin.com", "stackoverflow.com", "wikipedia.org", "adobe.com",
    # Media & Streaming
    "netflix.com", "crunchyroll.com", "spotify.com", "twitch.tv", "hulu.com",
    "bbc.co.uk", "nytimes.com", "cnn.com",
    # Education & Gov
    "stanford.edu", "mit.edu", "nasa.gov", "ucd.ie", "universityofgalway.ie",
    "indrashil-university.edu.in",
    # Finance
    "paypal.com", "chase.com", "bankofamerica.com", "stripe.com"
]

# Keywords often associated with phishing campaigns
RISK_KEYWORDS = ["secure", "login", "verify", "update", "bank", "account", "signin", "recovery", "gift", "confirm"]

# High-risk Top Level Domains
RISKY_TLDS = ['.xyz', '.top', '.club', '.tk', '.gq', '.cn', '.vip', '.work', '.info']