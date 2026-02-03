"""
Query Template Matcher Module
Matches user queries to predefined SQL templates and extracts parameters
"""
import re
import csv
import logging
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime, timedelta
from difflib import SequenceMatcher
import dateparser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryTemplate:
    """Represents a query template with parameter extraction"""
    
    def __init__(self, user_query: str, true_sql: str):
        self.user_query = user_query
        self.true_sql = true_sql
        self.parameters = self._extract_parameters()
        self.normalized_query = self._normalize_query(user_query)
    
    def _extract_parameters(self) -> List[Dict[str, Any]]:
        """Extract parameters from the SQL query"""
        parameters = []
        
        # Extract date parameters
        date_pattern = r"'(\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2}:\d{2})?)'"
        for match in re.finditer(date_pattern, self.true_sql):
            param_value = match.group(1)
            param_position = match.start()
            parameters.append({
                'type': 'date',
                'value': param_value,
                'position': param_position,
                'placeholder': f'{{date_{len(parameters)}}}'
            })
        
        # Extract interval parameters
        interval_pattern = r"INTERVAL\s+'(\d+)\s+(\w+)'"
        for match in re.finditer(interval_pattern, self.true_sql):
            param_value = f"{match.group(1)} {match.group(2)}"
            param_position = match.start()
            parameters.append({
                'type': 'interval',
                'value': param_value,
                'position': param_position,
                'placeholder': f'{{interval_{len(parameters)}}}'
            })
        
        # Extract number parameters (like LIMIT)
        limit_pattern = r'\bLIMIT\s+(\d+)'
        for match in re.finditer(limit_pattern, self.true_sql, re.IGNORECASE):
            param_value = match.group(1)
            param_position = match.start()
            parameters.append({
                'type': 'limit',
                'value': param_value,
                'position': param_position,
                'placeholder': f'{{limit_{len(parameters)}}}'
            })
        
        return parameters
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for better matching"""
        # Remove dates and numbers
        normalized = re.sub(r'\d{4}-\d{2}-\d{2}', '{DATE}', query)
        normalized = re.sub(r'\d+', '{NUM}', normalized)
        
        # Convert to lowercase and remove extra spaces
        normalized = ' '.join(normalized.lower().split())
        
        return normalized
    
    def get_template_sql(self) -> str:
        """Get SQL with placeholders instead of actual values"""
        template_sql = self.true_sql
        # Replace parameters with placeholders (in reverse order to preserve positions)
        for param in sorted(self.parameters, key=lambda x: x['position'], reverse=True):
            if param['type'] == 'date':
                template_sql = template_sql.replace(f"'{param['value']}'", param['placeholder'])
            elif param['type'] == 'interval':
                parts = param['value'].split()
                template_sql = template_sql.replace(f"INTERVAL '{parts[0]} {parts[1]}'", param['placeholder'])
            elif param['type'] == 'limit':
                template_sql = re.sub(
                    rf'\bLIMIT\s+{param["value"]}\b',
                    param['placeholder'],
                    template_sql,
                    flags=re.IGNORECASE
                )
        
        return template_sql
    
    def similarity_score(self, user_query: str) -> float:
        """Calculate similarity score with user query"""
        normalized_user = self._normalize_query(user_query)
        return SequenceMatcher(None, self.normalized_query, normalized_user).ratio()


class QueryTemplateMatcher:
    """Matches user queries to predefined templates"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.templates: List[QueryTemplate] = []
        self._load_templates()
    
    def _load_templates(self):
        """Load templates from CSV file"""
        try:
            self.templates = []  # Clear existing templates
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    user_query = row.get('user_query', '').strip()
                    true_sql = row.get('true_sql', '').strip()
                    
                    # Ignore few-shot columns and feedback column (only need user_query and true_sql)
                    if user_query and true_sql:
                        template = QueryTemplate(user_query, true_sql)
                        self.templates.append(template)
            
            logger.info(f"Loaded {len(self.templates)} query templates from {self.csv_path}")
        except Exception as e:
            logger.error(f"Error loading templates: {str(e)}")
            raise
    
    def reload_templates(self):
        """Reload templates from CSV file"""
        self._load_templates()
    
    def _extract_dates_from_query(self, query: str) -> List[str]:
        """Extract dates from user query using dateparser"""
        dates = []
        
        # Try to find explicit dates first (YYYY-MM-DD format)
        explicit_dates = re.findall(r'\d{4}-\d{2}-\d{2}', query)
        dates.extend(explicit_dates)
        
        # Parse relative dates
        relative_patterns = [
            (r'yesterday', lambda: (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')),
            (r'day before yesterday', lambda: (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')),
            (r'last\s+two\s+days', lambda: (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')),
            (r'last\s+week', lambda: (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')),
            (r'last\s+month', lambda: (datetime.now().replace(day=1) - timedelta(days=1)).strftime('%Y-%m')),
        ]
        
        for pattern, date_func in relative_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                dates.append(date_func())
        
        # If no dates found yet, try dateparser on phrases
        if not dates:
            words = query.lower().split()
            for i in range(len(words)):
                for j in range(i+1, min(i+6, len(words)+1)):
                    phrase = ' '.join(words[i:j])
                    parsed_date = dateparser.parse(phrase)
                    if parsed_date:
                        dates.append(parsed_date.strftime('%Y-%m-%d'))
                        break
        
        return dates
    
    def _extract_numbers_from_query(self, query: str) -> List[int]:
        """Extract numbers from user query"""
        return [int(n) for n in re.findall(r'\b(\d+)\b', query)]
    
    def _substitute_parameters(self, template: QueryTemplate, user_query: str) -> str:
        """Substitute parameters from user query into template SQL"""
        sql = template.true_sql
        
        # Extract dates from user query
        user_dates = self._extract_dates_from_query(user_query)
        
        # Extract numbers from user query
        user_numbers = self._extract_numbers_from_query(user_query)
        
        # Sort parameters by position
        sorted_params = sorted(template.parameters, key=lambda x: x['position'])
        
        date_idx = 0
        number_idx = 0
        
        # Replace parameters (in reverse order to preserve positions)
        for param in reversed(sorted_params):
            if param['type'] == 'date':
                if date_idx < len(user_dates):
                    new_value = user_dates[date_idx]
                    # Preserve the time format if it exists in original
                    if ' ' in param['value']:
                        # Has time component
                        time_part = param['value'].split()[1]
                        new_value = f"{new_value} {time_part}"
                    
                    sql = sql.replace(f"'{param['value']}'", f"'{new_value}'", 1)
                    date_idx += 1
            
            elif param['type'] == 'interval':
                # Keep interval as is (already parsed from template)
                pass
            
            elif param['type'] == 'limit':
                if number_idx < len(user_numbers):
                    new_value = str(user_numbers[number_idx])
                    sql = re.sub(
                        rf'\bLIMIT\s+{param["value"]}\b',
                        f'LIMIT {new_value}',
                        sql,
                        count=1,
                        flags=re.IGNORECASE
                    )
                    number_idx += 1
        
        return sql
    
    def match_query(self, user_query: str, threshold: float = 0.6) -> Optional[Tuple[QueryTemplate, float, str]]:
        """
        Match user query to a template
        Returns: (template, similarity_score, generated_sql) or None
        """
        if not self.templates:
            return None
        
        # Find best matching template
        best_match = None
        best_score = 0.0
        
        for template in self.templates:
            score = template.similarity_score(user_query)
            if score > best_score:
                best_score = score
                best_match = template
        
        # Check if score meets threshold
        if best_score >= threshold and best_match:
            try:
                # Generate SQL with substituted parameters
                generated_sql = self._substitute_parameters(best_match, user_query)
                logger.info(f"Matched query with score {best_score:.2f}: {best_match.user_query}")
                return best_match, best_score, generated_sql
            except Exception as e:
                logger.error(f"Error substituting parameters: {str(e)}")
                return None
        
        logger.info(f"No match found. Best score: {best_score:.2f} (threshold: {threshold})")
        return None
    
    def get_all_templates(self) -> List[Dict[str, str]]:
        """Get all templates as a list of dicts"""
        return [
            {
                'user_query': t.user_query,
                'true_sql': t.true_sql,
                'normalized_query': t.normalized_query,
                'parameters': t.parameters
            }
            for t in self.templates
        ]


# Example usage
if __name__ == "__main__":
    matcher = QueryTemplateMatcher("salespoint_testing_samples_solved.csv")
    
    # Test queries
    test_queries = [
        "Get all prepaid activations from 2025-05-31 to 2025-06-31",
        "Get all prepaid activations from yesterday",
        "Get best performance dealers top 5",
        "Show me all HBB activations",
    ]
    
    for query in test_queries:
        print(f"\n\nTest Query: {query}")
        print("=" * 80)
        result = matcher.match_query(query)
        
        if result:
            template, score, sql = result
            print(f"✅ Match Found! (Score: {score:.2f})")
            print(f"Template: {template.user_query}")
            print(f"Generated SQL: {sql}")
        else:
            print("❌ No match found - will use PGAI API")

