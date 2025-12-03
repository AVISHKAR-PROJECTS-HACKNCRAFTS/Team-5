"""
NLP Service for FIR Generation System.
Uses transformer-based models for Named Entity Recognition.
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntities:
    """Data class for extracted entities."""
    persons: List[str]
    locations: List[str]
    dates: List[str]
    times: List[str]
    organizations: List[str]
    money: List[str]


class NLPService:
    """
    NLP Service using transformer-based models for entity extraction.
    Supports both spaCy transformers and HuggingFace models.
    """
    
    def __init__(self, use_transformers: bool = True):
        """
        Initialize NLP Service.
        
        Args:
            use_transformers: If True, use HuggingFace transformers; else use spaCy
        """
        self.use_transformers = use_transformers
        self.ner_pipeline = None
        self.spacy_nlp = None
        self.use_regex_only = False
        self._load_models()
    
    def _load_models(self):
        """Load NER models based on configuration."""
        if self.use_transformers:
            self._load_transformer_model()
        else:
            self._load_spacy_model()
    
    def _load_transformer_model(self):
        """Load HuggingFace transformer NER model."""
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
            
            logger.info("Loading transformer NER model...")
            
            # Use a well-performing NER model
            model_name = "dslim/bert-base-NER"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple"
            )
            
            logger.info("Transformer NER model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load transformer model: {e}")
            logger.info("Falling back to spaCy model")
            self.use_transformers = False
            self._load_spacy_model()
    
    def _load_spacy_model(self):
        """Load spaCy model as fallback."""
        try:
            import spacy
            
            # Try to load transformer model first
            try:
                self.spacy_nlp = spacy.load("en_core_web_trf")
                logger.info("Loaded spaCy transformer model (en_core_web_trf)")
            except OSError:
                # Fall back to large model
                try:
                    self.spacy_nlp = spacy.load("en_core_web_lg")
                    logger.info("Loaded spaCy large model (en_core_web_lg)")
                except OSError:
                    # Fall back to small model
                    self.spacy_nlp = spacy.load("en_core_web_sm")
                    logger.info("Loaded spaCy small model (en_core_web_sm)")
                    
        except Exception as e:
            logger.warning(f"Failed to load spaCy model: {e}")
            logger.info("Using regex-based entity extraction (no ML model)")
            self.use_regex_only = True
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text for entity extraction
            
        Returns:
            Dictionary containing extracted entities by category
        """
        if self.use_transformers and self.ner_pipeline:
            return self._extract_with_transformers(text)
        elif self.spacy_nlp:
            return self._extract_with_spacy(text)
        else:
            return self._extract_with_regex(text)
    
    def _extract_with_transformers(self, text: str) -> Dict[str, Any]:
        """Extract entities using HuggingFace transformers."""
        entities = {
            "persons": [],
            "locations": [],
            "dates": [],
            "times": [],
            "organizations": [],
            "money": []
        }
        
        try:
            # Run NER pipeline
            ner_results = self.ner_pipeline(text)
            
            for entity in ner_results:
                entity_text = entity['word'].strip()
                entity_label = entity['entity_group']
                score = entity['score']
                
                # Filter low confidence entities
                if score < 0.5:
                    continue
                
                # Clean up tokenization artifacts
                entity_text = entity_text.replace('##', '')
                
                if entity_label == "PER":
                    if entity_text and len(entity_text) > 1:
                        entities["persons"].append(entity_text)
                elif entity_label == "LOC":
                    if entity_text and len(entity_text) > 1:
                        entities["locations"].append(entity_text)
                elif entity_label == "ORG":
                    if entity_text and len(entity_text) > 1:
                        entities["organizations"].append(entity_text)
            
            # Extract dates and times using regex (more reliable)
            entities["dates"] = self._extract_dates(text)
            entities["times"] = self._extract_times(text)
            entities["money"] = self._extract_money(text)
            
            # Also try to extract Indian locations using pattern matching
            indian_locations = self._extract_indian_locations(text)
            entities["locations"].extend(indian_locations)
            
            # Deduplicate
            for key in entities:
                entities[key] = list(dict.fromkeys(entities[key]))
            
        except Exception as e:
            logger.error(f"Error in transformer entity extraction: {e}")
            # Fall back to spaCy
            return self._extract_with_spacy(text)
        
        return entities
    
    def _extract_with_spacy(self, text: str) -> Dict[str, Any]:
        """Extract entities using spaCy."""
        entities = {
            "persons": [],
            "locations": [],
            "dates": [],
            "times": [],
            "organizations": [],
            "money": []
        }
        
        try:
            doc = self.spacy_nlp(text)
            
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    entities["persons"].append(ent.text)
                elif ent.label_ in ["GPE", "LOC", "FAC"]:
                    entities["locations"].append(ent.text)
                elif ent.label_ == "DATE":
                    entities["dates"].append(ent.text)
                elif ent.label_ == "TIME":
                    entities["times"].append(ent.text)
                elif ent.label_ == "ORG":
                    entities["organizations"].append(ent.text)
                elif ent.label_ == "MONEY":
                    entities["money"].append(ent.text)
            
            # Supplement with regex-based extraction
            entities["dates"].extend(self._extract_dates(text))
            entities["times"].extend(self._extract_times(text))
            
            # Extract Indian locations
            indian_locations = self._extract_indian_locations(text)
            entities["locations"].extend(indian_locations)
            
            # Deduplicate
            for key in entities:
                entities[key] = list(dict.fromkeys(entities[key]))
                
        except Exception as e:
            logger.error(f"Error in spaCy entity extraction: {e}")
        
        return entities
    
    def _extract_with_regex(self, text: str) -> Dict[str, Any]:
        """Extract entities using regex patterns only (fallback when no ML model available)."""
        entities = {
            "persons": [],
            "locations": [],
            "dates": [],
            "times": [],
            "organizations": [],
            "money": []
        }
        
        try:
            # Extract dates and times using regex
            entities["dates"] = self._extract_dates(text)
            entities["times"] = self._extract_times(text)
            entities["money"] = self._extract_money(text)
            
            # Extract Indian locations using pattern matching
            entities["locations"] = self._extract_indian_locations(text)
            
            # Extract person names using common patterns
            entities["persons"] = self._extract_persons_regex(text)
            
            # Extract organizations using common patterns
            entities["organizations"] = self._extract_organizations_regex(text)
            
            # Deduplicate
            for key in entities:
                entities[key] = list(dict.fromkeys(entities[key]))
                
        except Exception as e:
            logger.error(f"Error in regex entity extraction: {e}")
        
        return entities
    
    def _extract_persons_regex(self, text: str) -> List[str]:
        """Extract person names using regex patterns."""
        persons = []
        
        # Pattern for names following common prefixes
        name_patterns = [
            r'(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Shri|Smt\.?|Kumari)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})',
            r'(?:named?|called?|accused|complainant|victim)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})',
            r'(?:son|daughter|wife|husband)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})',
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, text)
            persons.extend(matches)
        
        return persons
    
    def _extract_organizations_regex(self, text: str) -> List[str]:
        """Extract organization names using regex patterns."""
        orgs = []
        
        # Common organization suffixes
        org_patterns = [
            r'\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\s+(?:Bank|Police|Station|Hospital|College|University|Company|Ltd|Pvt|Inc|Corp|Limited|Private))\b',
            r'\b(State\s+Bank\s+of\s+[A-Za-z]+)\b',
            r'\b(Reserve\s+Bank\s+of\s+India|RBI)\b',
            r'\b([A-Z]+\s+Bank)\b',
        ]
        
        for pattern in org_patterns:
            matches = re.findall(pattern, text)
            orgs.extend(matches)
        
        return orgs
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates using comprehensive regex patterns."""
        date_patterns = [
            # DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
            r'\b(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b',
            # YYYY/MM/DD or YYYY-MM-DD
            r'\b(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})\b',
            # Month name patterns: 1st January 2024, January 1, 2024, etc.
            r'\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b',
            r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})\b',
            # Short month: 1st Jan 2024, Jan 1, 2024
            r'\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})\b',
            r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})\b',
            # Relative dates
            r'\b(yesterday|today|last\s+(?:week|month|year|night|evening|morning))\b',
            r'\b((?:on\s+)?(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday))\b',
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)
        
        return list(dict.fromkeys(dates))
    
    def _extract_times(self, text: str) -> List[str]:
        """Extract times using comprehensive regex patterns."""
        time_patterns = [
            # 12-hour format: 3:30 PM, 3:30PM, 3.30 PM
            r'\b(\d{1,2}[:.]\d{2}\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.))\b',
            # 24-hour format: 15:30, 15.30
            r'\b(\d{1,2}[:.]\d{2}(?:\s*(?:hrs?|hours?))?)\b',
            # Written times: 3 PM, 3PM, 3 o'clock
            r'\b(\d{1,2}\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.|o\'?clock))\b',
            # Time ranges
            r'\b(\d{1,2}[:.]\d{2}\s*(?:to|-)\s*\d{1,2}[:.]\d{2}\s*(?:AM|PM|am|pm)?)\b',
            # Relative times
            r'\b(morning|afternoon|evening|night|noon|midnight)\b',
            # Around/about time
            r'\b(around\s+\d{1,2}(?:[:.]\d{2})?\s*(?:AM|PM|am|pm)?)\b',
            r'\b(about\s+\d{1,2}(?:[:.]\d{2})?\s*(?:AM|PM|am|pm)?)\b',
        ]
        
        times = []
        for pattern in time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            times.extend(matches)
        
        return list(dict.fromkeys(times))
    
    def _extract_money(self, text: str) -> List[str]:
        """Extract monetary amounts."""
        money_patterns = [
            # Rs/Rs./INR/₹ followed by number (with or without space, commas)
            r'(Rs\.?\s*\d[\d,]*(?:\.\d{2})?)',
            r'(INR\s*\d[\d,]*(?:\.\d{2})?)',
            r'(₹\s*\d[\d,]*(?:\.\d{2})?)',
            # Number followed by rupees
            r'(\d[\d,]*(?:\.\d{2})?\s*rupees?)',
            # Lakh/Crore format
            r'(Rs\.?\s*\d+(?:\.\d+)?\s*(?:lakh|lac|crore|cr)s?)',
            r'(\d+(?:\.\d+)?\s*(?:lakh|lac|crore|cr)s?)',
            # USD/Dollar
            r'(\$\s*\d[\d,]*(?:\.\d{2})?)',
            r'(USD\s*\d[\d,]*(?:\.\d{2})?)',
        ]
        
        money = []
        for pattern in money_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for m in matches:
                if m.strip():
                    money.append(m.strip())
        
        return list(dict.fromkeys(money))
    
    def _extract_indian_locations(self, text: str) -> List[str]:
        """Extract Indian location names using pattern matching."""
        # Common Indian cities and states
        indian_locations = [
            # Metro cities
            "Mumbai", "Delhi", "Bangalore", "Bengaluru", "Chennai", "Kolkata", 
            "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Kanpur",
            "Nagpur", "Indore", "Thane", "Bhopal", "Visakhapatnam", "Patna",
            "Vadodara", "Ghaziabad", "Ludhiana", "Agra", "Nashik", "Faridabad",
            "Meerut", "Rajkot", "Varanasi", "Srinagar", "Aurangabad", "Dhanbad",
            "Amritsar", "Allahabad", "Ranchi", "Howrah", "Coimbatore", "Jabalpur",
            "Gwalior", "Vijayawada", "Jodhpur", "Madurai", "Raipur", "Kota",
            "Chandigarh", "Guwahati", "Solapur", "Hubli", "Tiruchirappalli",
            "Bareilly", "Mysore", "Mysuru", "Tiruppur", "Gurgaon", "Gurugram",
            "Noida", "Greater Noida", "Dwarka",
            # States
            "Maharashtra", "Karnataka", "Tamil Nadu", "Telangana", "Andhra Pradesh",
            "Gujarat", "Rajasthan", "Uttar Pradesh", "Madhya Pradesh", "West Bengal",
            "Bihar", "Punjab", "Haryana", "Kerala", "Odisha", "Jharkhand",
            "Assam", "Chhattisgarh", "Uttarakhand", "Himachal Pradesh", "Tripura",
            "Meghalaya", "Manipur", "Nagaland", "Goa", "Arunachal Pradesh",
            "Mizoram", "Sikkim"
        ]
        
        found_locations = []
        text_lower = text.lower()
        
        for location in indian_locations:
            if location.lower() in text_lower:
                # Find the actual case from text
                pattern = re.compile(re.escape(location), re.IGNORECASE)
                match = pattern.search(text)
                if match:
                    found_locations.append(match.group())
        
        return found_locations


# Singleton instance
_nlp_service: Optional[NLPService] = None


def get_nlp_service(use_transformers: bool = True) -> NLPService:
    """
    Get or create NLP service singleton.
    
    Args:
        use_transformers: Whether to use transformer models
        
    Returns:
        NLPService instance
    """
    global _nlp_service
    if _nlp_service is None:
        _nlp_service = NLPService(use_transformers=use_transformers)
    return _nlp_service
