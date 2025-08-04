"""
Migration Guide for Unified Logging System.

This module provides utilities and guidance for migrating existing modules
to use the new unified logging system.
"""

import re
import ast
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class LoggingMigrationHelper:
    """Helper class for migrating existing logging code."""
    
    # Common logging import patterns to replace
    OLD_IMPORTS = [
        r'from core\.logging_setup import.*',
        r'from \.core\.logging_setup import.*',
        r'from infrastructure\.logging import.*',
        r'from \.infrastructure\.logging import.*',
        r'import logging',
    ]
    
    # New import statements
    NEW_IMPORTS = [
        'from .core.unified_logging import get_logger, log_structured, log_api_call',
        'from .core.logging_config import LoggingPatterns, StandardLogMessages',
    ]
    
    # Common logger instantiation patterns
    OLD_LOGGER_PATTERNS = [
        r'logger = logging\.getLogger\(__name__\)',
        r'logger = logging\.getLogger\([\'"].*[\'"]\)',
        r'logger = get_logger\(__name__\)',
        r'logger = get_logger\([\'"].*[\'"]\)',
    ]
    
    def __init__(self):
        self.migration_stats = {
            'files_processed': 0,
            'imports_updated': 0,
            'logger_calls_updated': 0,
            'print_statements_found': 0,
            'issues_found': [],
        }
    
    def analyze_file(self, file_path: Path) -> Dict[str, any]:
        """
        Analyze a Python file for logging migration opportunities.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Analysis results dictionary
        """
        if not file_path.exists() or file_path.suffix != '.py':
            return {'error': 'File not found or not a Python file'}
        
        content = file_path.read_text(encoding='utf-8')
        
        analysis = {
            'file_path': str(file_path),
            'old_imports': [],
            'old_logger_instances': [],
            'print_statements': [],
            'logging_calls': [],
            'recommendations': [],
        }
        
        # Find old import patterns
        for pattern in self.OLD_IMPORTS:
            matches = re.findall(pattern, content, re.MULTILINE)
            analysis['old_imports'].extend(matches)
        
        # Find old logger instantiation patterns
        for pattern in self.OLD_LOGGER_PATTERNS:
            matches = re.findall(pattern, content, re.MULTILINE)
            analysis['old_logger_instances'].extend(matches)
        
        # Find print statements that should be logging
        print_pattern = r'print\s*\([^)]*\)'
        print_matches = re.findall(print_pattern, content)
        analysis['print_statements'] = print_matches
        
        # Find logging method calls
        logging_pattern = r'logger\.(debug|info|warning|error|critical|exception)\s*\([^)]*\)'
        logging_matches = re.findall(logging_pattern, content)
        analysis['logging_calls'] = logging_matches
        
        # Generate recommendations
        self._generate_recommendations(analysis)
        
        return analysis
    
    def _generate_recommendations(self, analysis: Dict[str, any]):
        """Generate migration recommendations based on analysis."""
        recommendations = []
        
        if analysis['old_imports']:
            recommendations.append(
                "Update imports to use unified logging system: "
                "from .core.unified_logging import get_logger, log_structured"
            )
        
        if analysis['old_logger_instances']:
            recommendations.append(
                "Update logger instantiation to use: "
                "logger = get_logger(create_logger_name(__name__))"
            )
        
        if analysis['print_statements']:
            recommendations.append(
                f"Found {len(analysis['print_statements'])} print statements that "
                "should potentially be converted to logging calls"
            )
        
        if analysis['logging_calls']:
            recommendations.append(
                f"Consider converting {len(analysis['logging_calls'])} logging calls "
                "to use structured logging for better consistency"
            )
        
        analysis['recommendations'] = recommendations
    
    def migrate_file(self, file_path: Path, dry_run: bool = True) -> Dict[str, any]:
        """
        Migrate a file to use the new logging system.
        
        Args:
            file_path: Path to the Python file
            dry_run: If True, only show what would be changed
            
        Returns:
            Migration results dictionary
        """
        if not file_path.exists() or file_path.suffix != '.py':
            return {'error': 'File not found or not a Python file'}
        
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        
        migration_result = {
            'file_path': str(file_path),
            'changes_made': [],
            'dry_run': dry_run,
            'success': False,
        }
        
        # Replace old imports
        changes_made = 0
        for old_pattern in self.OLD_IMPORTS:
            if re.search(old_pattern, content):
                # Remove old import
                content = re.sub(old_pattern, '', content, flags=re.MULTILINE)
                changes_made += 1
                migration_result['changes_made'].append(f"Removed old import: {old_pattern}")
        
        # Add new imports at the top (after existing imports)
        if changes_made > 0:
            # Find the best place to insert new imports
            lines = content.split('\n')
            insert_line = 0
            
            # Find last import line
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ')) and not line.strip().startswith('#'):
                    insert_line = i + 1
            
            # Insert new imports
            new_import_line = 'from .core.unified_logging import get_logger, log_structured'
            if new_import_line not in content:
                lines.insert(insert_line, new_import_line)
                content = '\n'.join(lines)
                migration_result['changes_made'].append("Added new unified logging imports")
        
        # Replace logger instantiation patterns
        for old_pattern in self.OLD_LOGGER_PATTERNS:
            new_logger_line = 'logger = get_logger(create_logger_name(__name__))'
            if re.search(old_pattern, content):
                content = re.sub(old_pattern, new_logger_line, content)
                migration_result['changes_made'].append(f"Updated logger instantiation")
        
        # If we have changes and not dry run, write the file
        if migration_result['changes_made'] and not dry_run:
            try:
                file_path.write_text(content, encoding='utf-8')
                migration_result['success'] = True
            except Exception as e:
                migration_result['error'] = f"Failed to write file: {e}"
        elif migration_result['changes_made']:
            migration_result['success'] = True
            migration_result['preview'] = content
        
        return migration_result
    
    def scan_directory(self, directory: Path, recursive: bool = True) -> List[Dict[str, any]]:
        """
        Scan a directory for files that need logging migration.
        
        Args:
            directory: Directory to scan
            recursive: Whether to scan subdirectories
            
        Returns:
            List of analysis results for each file
        """
        results = []
        
        if recursive:
            pattern = '**/*.py'
        else:
            pattern = '*.py'
        
        for py_file in directory.glob(pattern):
            # Skip __pycache__ and other irrelevant directories
            if '__pycache__' in str(py_file) or '.git' in str(py_file):
                continue
            
            analysis = self.analyze_file(py_file)
            if 'error' not in analysis:
                results.append(analysis)
        
        return results
    
    def generate_migration_report(self, analyses: List[Dict[str, any]]) -> str:
        """Generate a migration report from analysis results."""
        report_lines = []
        report_lines.append("# Logging Migration Report\n")
        
        total_files = len(analyses)
        files_needing_migration = 0
        total_print_statements = 0
        total_old_imports = 0
        
        for analysis in analyses:
            if (analysis['old_imports'] or 
                analysis['old_logger_instances'] or 
                analysis['print_statements']):
                files_needing_migration += 1
            
            total_print_statements += len(analysis['print_statements'])
            total_old_imports += len(analysis['old_imports'])
        
        # Summary
        report_lines.append(f"## Summary\n")
        report_lines.append(f"- Total files analyzed: {total_files}")
        report_lines.append(f"- Files needing migration: {files_needing_migration}")
        report_lines.append(f"- Total old imports found: {total_old_imports}")
        report_lines.append(f"- Total print statements found: {total_print_statements}\n")
        
        # Detailed file analysis
        report_lines.append("## File Analysis\n")
        
        for analysis in analyses:
            if not (analysis['old_imports'] or 
                    analysis['old_logger_instances'] or 
                    analysis['print_statements']):
                continue
            
            report_lines.append(f"### {analysis['file_path']}\n")
            
            if analysis['old_imports']:
                report_lines.append("**Old imports found:**")
                for imp in analysis['old_imports']:
                    report_lines.append(f"- `{imp}`")
                report_lines.append("")
            
            if analysis['old_logger_instances']:
                report_lines.append("**Old logger instances:**")
                for logger in analysis['old_logger_instances']:
                    report_lines.append(f"- `{logger}`")
                report_lines.append("")
            
            if analysis['print_statements']:
                report_lines.append(f"**Print statements found:** {len(analysis['print_statements'])}")
                for i, stmt in enumerate(analysis['print_statements'][:5]):  # Show first 5
                    report_lines.append(f"- `{stmt}`")
                if len(analysis['print_statements']) > 5:
                    report_lines.append(f"- ... and {len(analysis['print_statements']) - 5} more")
                report_lines.append("")
            
            if analysis['recommendations']:
                report_lines.append("**Recommendations:**")
                for rec in analysis['recommendations']:
                    report_lines.append(f"- {rec}")
                report_lines.append("")
        
        # Migration steps
        report_lines.append("## Migration Steps\n")
        report_lines.append("1. **Update imports**: Replace old logging imports with:")
        report_lines.append("   ```python")
        report_lines.append("   from .core.unified_logging import get_logger, log_structured")
        report_lines.append("   from .core.logging_config import LoggingPatterns, StandardLogMessages, create_logger_name")
        report_lines.append("   ```\n")
        
        report_lines.append("2. **Update logger instantiation**: Replace with:")
        report_lines.append("   ```python")
        report_lines.append("   logger = get_logger(create_logger_name(__name__))")
        report_lines.append("   ```\n")
        
        report_lines.append("3. **Use structured logging**: For important operations, consider:")
        report_lines.append("   ```python")
        report_lines.append("   log_structured('INFO', LoggingPatterns.COMPONENTS['AUDIO'], 'Processing file', file_path=path)")
        report_lines.append("   ```\n")
        
        report_lines.append("4. **Replace print statements**: Convert debug prints to logger.debug()")
        report_lines.append("5. **Use standard message templates**: Utilize StandardLogMessages for consistency")
        
        return '\n'.join(report_lines)


# Example migration patterns and code snippets
MIGRATION_EXAMPLES = {
    'basic_logger_setup': {
        'before': '''
import logging

logger = logging.getLogger(__name__)
        ''',
        'after': '''
from .core.unified_logging import get_logger
from .core.logging_config import create_logger_name

logger = get_logger(create_logger_name(__name__))
        '''
    },
    
    'structured_logging': {
        'before': '''
logger.info(f"Processing file {file_path} with size {file_size}")
        ''',
        'after': '''
log_structured(
    'INFO',
    LoggingPatterns.COMPONENTS['AUDIO'],
    'Processing file',
    file_path=file_path,
    file_size_mb=file_size
)
        '''
    },
    
    'api_call_logging': {
        'before': '''
try:
    response = api_call()
    logger.info(f"API call successful: {response}")
except Exception as e:
    logger.error(f"API call failed: {e}")
        ''',
        'after': '''
start_time = time.time()
try:
    response = api_call()
    duration = time.time() - start_time
    log_api_call('MusicBrainz', 'search', 'artist', True, duration=duration)
except Exception as e:
    duration = time.time() - start_time
    log_api_call('MusicBrainz', 'search', 'artist', False, duration=duration)
        '''
    },
    
    'error_handling': {
        'before': '''
try:
    result = risky_operation()
except Exception as e:
    logger.error(f"Operation failed: {e}")
    raise
        ''',
        'after': '''
try:
    result = risky_operation()
except ValueError as e:
    logger.warning(f"Invalid input: {e}")
    return None
except Exception as e:
    logger.exception(f"Unexpected error: {e}")
    raise
        '''
    },
    
    'performance_logging': {
        'before': '''
start = time.time()
process_files(files)
end = time.time()
logger.info(f"Processed {len(files)} files in {end-start:.2f}s")
        ''',
        'after': '''
with LoggedOperation("file_processing", f"{len(files)} files", "Audio"):
    process_files(files)
        '''
    }
}


def demonstrate_migration():
    """Demonstrate the migration process."""
    print("=== Logging Migration Examples ===\n")
    
    for example_name, example in MIGRATION_EXAMPLES.items():
        print(f"## {example_name.replace('_', ' ').title()}\n")
        print("**Before:**")
        print("```python")
        print(example['before'].strip())
        print("```\n")
        print("**After:**")
        print("```python")
        print(example['after'].strip())
        print("```\n")


if __name__ == "__main__":
    demonstrate_migration()