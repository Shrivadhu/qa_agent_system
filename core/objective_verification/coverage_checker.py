import json
import re
from pathlib import Path
from typing import List, Dict, Set, Optional
from utils.embeddings import EmbeddingManager
from utils.logger import log
from models.schemas import CoverageReport


class CoverageChecker:
    """Check coverage of promised objectives in session."""
    
    def __init__(self):
        """Initialize coverage checker."""
        self.embedding_manager = EmbeddingManager()
        self.skill_taxonomy = self._load_skill_taxonomy()
        log.info(f"CoverageChecker initialized with {len(self.skill_taxonomy)} skills")
    
    def _load_skill_taxonomy(self) -> Dict[str, List[str]]:
        """
        Load skill taxonomy and expected keywords from config/skill_taxonomy.json,
        falling back to default taxonomy if file is missing or invalid.
        """
        taxonomy_path = Path(__file__).resolve().parents[2] / "config" / "skill_taxonomy.json"
        if taxonomy_path.exists():
            try:
                with open(taxonomy_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                log.info(f"Loaded skill taxonomy from {taxonomy_path}")
                return data
            except Exception as e:
                log.warning(f"Error loading skill taxonomy from {taxonomy_path}: {e}")

        return {
            "python_basics": [
                "variable", "function", "loop", "if statement", "data type",
                "string", "integer", "list", "dictionary", "print",
                "input", "output", "syntax", "indentation"
            ],
            "python_functions": [
                "function", "def", "parameter", "argument", "return",
                "call", "reusable", "signature", "scope", "lambda",
                "default parameter", "keyword argument", "output"
            ],
            "data_structures": [
                "array", "list", "tuple", "dictionary", "set",
                "stack", "queue", "tree", "graph", "hash table",
                "linked list", "complexity", "algorithm"
            ],
            "web_development": [
                "html", "css", "javascript", "dom", "http",
                "api", "rest", "json", "frontend", "backend",
                "server", "client", "request", "response"
            ],
            "machine_learning": [
                "model", "training", "prediction", "feature", "label",
                "supervised", "unsupervised", "neural network", "dataset",
                "accuracy", "loss", "optimization", "gradient"
            ]
        }
    
    @staticmethod
    def _stem(word: str) -> str:
        """Very light plural stripping so 'functions' matches 'function'."""
        return word[:-1] if len(word) > 3 and word.endswith('s') and not word.endswith('ss') else word

    def generate_expected_keywords(
        self,
        promised_skills: List[str],
        custom_skill_keywords: Optional[Dict[str, List[str]]] = None
    ) -> List[str]:
        """
        Generate expected keywords for promised skills.
        
        Implementation of FEAT-6.1.2:
        - Takes session agreement (promised skills)
        - Checks custom_skill_keywords if provided
        - Generates expected keywords for each skill using skill taxonomy
        - Returns expected keyword sets
        
        Args:
            promised_skills: List of skills promised in session
            custom_skill_keywords: Optional dictionary of skill to keywords mapping
            
        Returns:
            List of expected keywords
        """
        log.info(f"Generating expected keywords for {len(promised_skills)} skills")
        custom_map = custom_skill_keywords or {}
        expected_keywords = set()
        
        for skill in promised_skills:
            skill_normalized = skill.lower().replace(" ", "_")
            
            # 1. Check custom keywords first
            if skill in custom_map:
                expected_keywords.update(custom_map[skill])
                log.debug(f"Added {len(custom_map[skill])} custom keywords for skill: {skill}")
            elif skill_normalized in custom_map:
                expected_keywords.update(custom_map[skill_normalized])
                log.debug(f"Added {len(custom_map[skill_normalized])} custom keywords for skill: {skill}")
            # 2. Check taxonomy
            elif skill_normalized in self.skill_taxonomy:
                keywords = self.skill_taxonomy[skill_normalized]
                expected_keywords.update(keywords)
                log.debug(f"Added {len(keywords)} taxonomy keywords for skill: {skill}")
            elif skill.lower() in self.skill_taxonomy:
                keywords = self.skill_taxonomy[skill.lower()]
                expected_keywords.update(keywords)
                log.debug(f"Added {len(keywords)} taxonomy keywords for skill: {skill}")
            else:
                # If skill not in taxonomy, use skill name as keyword
                log.warning(f"Skill not in taxonomy: {skill}")
                expected_keywords.add(skill.lower().replace('_', ' '))
        
        result = list(expected_keywords)
        log.info(f"Generated {len(result)} expected keywords")
        
        return result
    
    def check_coverage(
        self,
        extracted_keywords: List[Dict[str, any]],
        expected_keywords: List[str],
        similarity_threshold: float = 0.7,
        transcript: Optional[str] = None
    ) -> CoverageReport:
        """
        Check objective coverage by comparing keywords.
        
        Implementation of FEAT-6.1.3:
        - Takes extracted_keywords and expected_keywords
        - For each expected keyword, checks if in extracted
        - If transcript is provided, checks for direct word/phrase presence in transcript
        - Performs semantic similarity matching
        - Calculates coverage percentage
        - Generates coverage report
        - Flags missing topics
        
        Args:
            extracted_keywords: Keywords extracted from transcript
            expected_keywords: Keywords expected based on promises
            similarity_threshold: Threshold for semantic similarity
            transcript: Optional full session transcript for direct verbatim match
            
        Returns:
            CoverageReport with coverage metrics
        """
        log.info("Checking objective coverage")
        
        # Extract keyword strings from extracted keywords
        extracted_keyword_strings = {
            kw['keyword'].lower() for kw in extracted_keywords
        }
        
        extracted_list = sorted(extracted_keyword_strings)
        extracted_words = {
            self._stem(w) for kw in extracted_keyword_strings for w in kw.split()
        }
        transcript_lower = transcript.lower() if transcript else ""

        # Track covered and missing topics
        covered_topics = []
        missing_topics = []
        
        for expected in expected_keywords:
            expected_lower = expected.lower()
            
            # Direct match against extracted keywords
            expected_words = {self._stem(w) for w in expected_lower.split()}
            if expected_lower in extracted_keyword_strings or (
                expected_words and expected_words <= extracted_words
            ):
                covered_topics.append(expected)
                continue

            # Direct verbatim / word-boundary match in transcript text
            if transcript_lower:
                pattern = r'\b' + re.escape(expected_lower) + r'\b'
                if re.search(pattern, transcript_lower):
                    covered_topics.append(expected)
                    log.debug(f"Direct transcript match: '{expected}' found in session transcript")
                    continue
            
            # Semantic similarity match
            found_similar = False
            matches = self.embedding_manager.semantic_search(
                expected_lower,
                extracted_list,
                threshold=similarity_threshold
            )
            if matches:
                covered_topics.append(expected)
                found_similar = True
                log.debug(
                    f"Semantic match: '{expected}' ~ '{matches[0]['text']}' "
                    f"(similarity: {matches[0]['score']:.2f})"
                )
            
            if not found_similar:
                missing_topics.append(expected)
        
        # Calculate coverage percentage
        total_expected = len(expected_keywords)
        total_covered = len(covered_topics)
        
        coverage_percentage = (
            (total_covered / total_expected * 100)
            if total_expected > 0
            else 0.0
        )
        
        report = CoverageReport(
            coverage_percentage=coverage_percentage,
            covered_topics=covered_topics,
            missing_topics=missing_topics,
            expected_keyword_count=total_expected,
            found_keyword_count=total_covered
        )
        
        log.info(
            f"Coverage check complete: {coverage_percentage:.1f}% "
            f"({total_covered}/{total_expected} topics covered)"
        )
        
        if missing_topics:
            log.warning(f"Missing {len(missing_topics)} topics: {missing_topics[:5]}")
        
        return report
    
    def calculate_scope_completeness_score(
        self,
        coverage_report: CoverageReport
    ) -> float:
        """
        Calculate normalized scope completeness score.
        
        Args:
            coverage_report: Coverage report from check_coverage
            
        Returns:
            Score from 0.0 to 1.0
        """
        return coverage_report.coverage_percentage / 100.0
