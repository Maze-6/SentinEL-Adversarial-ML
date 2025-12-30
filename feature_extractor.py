import re
import math
import requests
import dns.resolver
import ssl
import socket
import datetime
import whois
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import config

class FeatureExtractor:
    """
    Extracts 17 features from a given URL including Lexical, Network, and Host-based attributes.
    """

    @staticmethod
    def shannon_entropy(text: str) -> float:
        """Calculates randomness of the URL string."""
        if not text: return 0.0
        probabilities = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
        return -sum([p * math.log(p) / math.log(2.0) for p in probabilities])

    @staticmethod
    def get_domain_age(domain: str) -> int:
        """
        Fetches domain age in days. Returns -1 if WHOIS lookup fails.
        """
        try:
            w = whois.whois(domain)
            creation_date = w.creation_date
            
            # Handle cases where multiple dates are returned
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
                
            if isinstance(creation_date, (datetime.datetime, datetime.date)):
                if creation_date.tzinfo is not None:
                    creation_date = creation_date.replace(tzinfo=None)
                return (datetime.datetime.now() - creation_date).days
        except Exception:
            return -1 
        return -1

    @staticmethod
    def extract_features(url: str) -> list:
        # 1. Page Content Analysis
        has_form, has_pass, has_iframe = 0, 0, 0
        status_code = 0
        
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(url, timeout=3.0, headers=headers)
            status_code = response.status_code
            
            soup = BeautifulSoup(response.text, 'html.parser')
            if soup.find('form'): has_form = 1
            if soup.find('input', {'type': 'password'}): has_pass = 1
            if soup.find('iframe'): has_iframe = 1
        except:
            status_code = 404 

        # 2. DNS & Domain Age
        parsed = urlparse(url)
        domain = parsed.netloc.split(':')[0]
        dns_res = 1
        domain_age = -1
        
        try:
            dns.resolver.resolve(domain, 'A')
            domain_age = FeatureExtractor.get_domain_age(domain)
        except:
            dns_res = 0

        # 3. SSL Certificate Validity
        ssl_risk = 0
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False 
            with ctx.wrap_socket(socket.socket(), server_hostname=domain) as s:
                s.settimeout(2.0)
                s.connect((domain, 443))
                cert = s.getpeercert()
                not_after = datetime.datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                # Flag if cert expires within 30 days
                if (not_after - datetime.datetime.now()).days < 30: ssl_risk = 1
        except:
            ssl_risk = 1 

        # 4. Lexical Analysis
        is_risky_tld = 1 if any(domain.endswith(t) for t in config.RISKY_TLDS) else 0

        features = [
            len(url),                                     
            sum(c.isdigit() for c in url),                
            FeatureExtractor.shannon_entropy(url),        
            is_risky_tld,                                 
            1 if re.search(r'\d{1,3}\.\d{1,3}', url) else 0, 
            url.count('.') - 1,                           
            url.count('-'),                               
            1 if any(k in url for k in config.RISK_KEYWORDS) else 0 
        ]
        
        features.extend([
            dns_res,    
            domain_age, 
            300,        # Simulated expiry for latency optimization
            has_form,   
            has_pass,   
            has_iframe, 
            0.1,        
            status_code,
            ssl_risk    
        ])
        
        return features

    @staticmethod
    def get_feature_names():
        return ["Length", "Digits", "Entropy", "Risky_TLD", "IP_Usage", "Subdomains", "Hyphens", "Keywords",
                "DNS_Rec", "Domain_Age", "Expiry", "Has_Form", "Pass_Field", "IFrame", "Link_Ratio", "HTTP_Code", "SSL_Risk"]