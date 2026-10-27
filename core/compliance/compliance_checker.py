"""Compliance checking module for session quality assurance."""

import re
from typing import List, Tuple
from utils.logger import log


class ComplianceChecker:
    """Checks session transcripts for policy and compliance violations."""

    def __init__(self, min_word_count: int = 50):
        """Initialize compliance checker."""
        self.min_word_count = min_word_count
        log.info("ComplianceChecker initialized")

    def check_compliance(self, transcript: str) -> Tuple[float, List[str]]:
        """
        Check transcript for compliance violations.

        Flags:
        - Short transcripts (< 50 words)
        - Shared emails
        - Shared phone numbers
        - WhatsApp / Telegram mentions
        - Off-platform payment talk
        - Abusive / inappropriate language

        Args:
            transcript: Full session transcript text

        Returns:
            Tuple of (compliance_score: float, violations: List[str])
        """
        if not transcript or not transcript.strip():
            return 0.0, ["empty or missing transcript"]

        text_lower = transcript.lower()
        violations: List[str] = []
        penalty = 0.0

        # 1. Transcript length check
        words = transcript.split()
        if len(words) < self.min_word_count:
            violations.append(f"short transcript ({len(words)} words, minimum required is {self.min_word_count})")
            penalty += 0.35

        # 2. Shared email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, transcript)
        if emails:
            violations.append("shared email address")
            penalty += 0.30

        # 3. Shared phone numbers (e.g., +1-555-123-4567, 555-123-4567, 10-12 consecutive digits)
        phone_pattern = r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|\b\d{10,12}\b'
        phones = re.findall(phone_pattern, transcript)
        if phones:
            violations.append("shared phone number")
            penalty += 0.30

        # 4. WhatsApp / Telegram / external messaging mentions
        messaging_pattern = r'\b(?:whatsapp|telegram|wa\.me|t\.me|viber|wechat|signal\s+app)\b'
        if re.search(messaging_pattern, text_lower):
            violations.append("external contact sharing (WhatsApp/Telegram)")
            penalty += 0.30

        # 5. Off-platform payment talk
        payment_pattern = (
            r'\b(?:paypal|venmo|cashapp|cash\s+app|zelle|crypto|bitcoin|btc|eth|usdt|'
            r'wire\s+transfer|bank\s+transfer|pay\s+(?:me\s+)?(?:directly|outside|offline)|'
            r'send\s+(?:money|payment)\s+to)\b'
        )
        if re.search(payment_pattern, text_lower):
            violations.append("off-platform payment discussion")
            penalty += 0.40

        # 6. Abusive or inappropriate language
        abusive_pattern = r'\b(?:idiot|stupid|moron|dumb|shut\s+up|hate\s+you|bastard|scam(?:mer)?|bitch|asshole|bullshit|trash|loser)\b'
        if re.search(abusive_pattern, text_lower):
            violations.append("abusive or inappropriate language")
            penalty += 0.40

        # Calculate final compliance score
        compliance_score = max(0.0, min(1.0, 1.0 - penalty))

        if violations:
            log.warning(f"Compliance violations detected: {violations} (score: {compliance_score:.2f})")
        else:
            log.debug("Session is fully compliant")

        return compliance_score, violations
