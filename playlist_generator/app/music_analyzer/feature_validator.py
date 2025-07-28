#!/usr/bin/env python3
"""
Feature validation for audio analysis results.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class FeatureValidator:
    """Validate extracted audio features for quality and completeness."""
    
    def __init__(self):
        """Initialize the feature validator."""
        self.required_fields = [
            'bpm', 'key', 'mode', 'danceability', 'energy', 'valence',
            'acousticness', 'instrumentalness', 'liveness', 'speechiness'
        ]
        
        self.optional_fields = [
            'tempo', 'loudness', 'key_confidence', 'mode_confidence',
            'onset_rate', 'zcr', 'spectral_centroid', 'spectral_rolloff',
            'spectral_flatness', 'mfcc', 'chroma', 'spectral_contrast'
        ]
    
    def validate_essential_fields(self, features: Dict[str, Any]) -> bool:
        """Validate that essential fields are present and valid.
        
        Args:
            features (Dict[str, Any]): Extracted features.
            
        Returns:
            bool: True if essential fields are valid, False otherwise.
        """
        try:
            # Check for required fields
            for field in self.required_fields:
                if field not in features:
                    logger.warning(f"Missing required field: {field}")
                    return False
                
                value = features[field]
                if value is None:
                    logger.warning(f"Required field {field} is None")
                    return False
                
                # Validate numeric fields
                if isinstance(value, (int, float, np.number)):
                    if np.isnan(value) or np.isinf(value):
                        logger.warning(f"Required field {field} has invalid value: {value}")
                        return False
                
                # Validate string fields
                elif isinstance(value, str):
                    if not value.strip():
                        logger.warning(f"Required field {field} is empty string")
                        return False
            
            # Validate BPM range
            bpm = features.get('bpm')
            if bpm is not None and (bpm < 20 or bpm > 300):
                logger.warning(f"BPM value {bpm} is outside valid range (20-300)")
                return False
            
            # Validate key values
            key = features.get('key')
            if key is not None and not isinstance(key, str):
                logger.warning(f"Key value {key} is not a string")
                return False
            
            # Validate mode values
            mode = features.get('mode')
            if mode is not None and mode not in ['major', 'minor']:
                logger.warning(f"Mode value {mode} is not valid")
                return False
            
            # Validate percentage fields (0-1 range)
            percentage_fields = ['danceability', 'energy', 'valence', 'acousticness', 
                               'instrumentalness', 'liveness', 'speechiness']
            for field in percentage_fields:
                value = features.get(field)
                if value is not None and (value < 0 or value > 1):
                    logger.warning(f"{field} value {value} is outside valid range (0-1)")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating essential fields: {e}")
            return False
    
    def validate_optional_fields(self, features: Dict[str, Any]) -> bool:
        """Validate optional fields if present.
        
        Args:
            features (Dict[str, Any]): Extracted features.
            
        Returns:
            bool: True if optional fields are valid, False otherwise.
        """
        try:
            for field in self.optional_fields:
                if field in features:
                    value = features[field]
                    
                    if value is not None:
                        # Validate numeric fields
                        if isinstance(value, (int, float, np.number)):
                            if np.isnan(value) or np.isinf(value):
                                logger.warning(f"Optional field {field} has invalid value: {value}")
                                return False
                        
                        # Validate array fields
                        elif isinstance(value, (list, np.ndarray)):
                            if len(value) == 0:
                                logger.warning(f"Optional field {field} is empty array")
                                return False
                            
                            # Check for NaN or inf values in arrays
                            if isinstance(value, np.ndarray):
                                if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                                    logger.warning(f"Optional field {field} contains invalid values")
                                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating optional fields: {e}")
            return False
    
    def validate_feature_quality(self, features: Dict[str, Any]) -> bool:
        """Validate overall feature quality and consistency.
        
        Args:
            features (Dict[str, Any]): Extracted features.
            
        Returns:
            bool: True if features meet quality standards, False otherwise.
        """
        try:
            # Check for reasonable value ranges
            quality_checks = [
                # BPM should be reasonable for music
                ('bpm', lambda x: 20 <= x <= 300 if x is not None else True),
                
                # Loudness should be reasonable
                ('loudness', lambda x: -60 <= x <= 0 if x is not None else True),
                
                # Spectral centroid should be positive
                ('spectral_centroid', lambda x: x > 0 if x is not None else True),
                
                # Spectral rolloff should be positive
                ('spectral_rolloff', lambda x: x > 0 if x is not None else True),
                
                # MFCC should have reasonable number of coefficients
                ('mfcc', lambda x: len(x) == 13 if isinstance(x, (list, np.ndarray)) else True),
                
                # Chroma should have 12 dimensions
                ('chroma', lambda x: len(x) == 12 if isinstance(x, (list, np.ndarray)) else True),
            ]
            
            for field, check_func in quality_checks:
                if field in features:
                    value = features[field]
                    if not check_func(value):
                        logger.warning(f"Quality check failed for {field}: {value}")
                        return False
            
            # Check for consistency between related features
            if 'bpm' in features and 'tempo' in features:
                bpm = features['bpm']
                tempo = features['tempo']
                if bpm is not None and tempo is not None:
                    # BPM and tempo should be similar (within 5 BPM)
                    if abs(bpm - tempo) > 5:
                        logger.warning(f"BPM ({bpm}) and tempo ({tempo}) are inconsistent")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating feature quality: {e}")
            return False
    
    def is_valid_feature_set(self, features: Dict[str, Any]) -> bool:
        """Check if a feature set is valid for playlist generation.
        
        Args:
            features (Dict[str, Any]): Extracted features.
            
        Returns:
            bool: True if features are valid for playlisting, False otherwise.
        """
        try:
            # Must have essential fields
            if not self.validate_essential_fields(features):
                return False
            
            # Optional fields should be valid if present
            if not self.validate_optional_fields(features):
                return False
            
            # Overall quality should be good
            if not self.validate_feature_quality(features):
                return False
            
            # Must have at least some basic musical features
            basic_features = ['bpm', 'key', 'mode', 'danceability', 'energy']
            basic_feature_count = sum(1 for f in basic_features if f in features and features[f] is not None)
            
            if basic_feature_count < 3:
                logger.warning(f"Insufficient basic features: {basic_feature_count}/5")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking feature set validity: {e}")
            return False
    
    def get_validation_report(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a detailed validation report for features.
        
        Args:
            features (Dict[str, Any]): Extracted features.
            
        Returns:
            Dict[str, Any]: Validation report with details.
        """
        report = {
            'valid': False,
            'essential_fields_valid': False,
            'optional_fields_valid': False,
            'quality_valid': False,
            'missing_fields': [],
            'invalid_fields': [],
            'quality_issues': []
        }
        
        try:
            # Check essential fields
            report['essential_fields_valid'] = self.validate_essential_fields(features)
            
            # Check optional fields
            report['optional_fields_valid'] = self.validate_optional_fields(features)
            
            # Check quality
            report['quality_valid'] = self.validate_feature_quality(features)
            
            # Find missing required fields
            for field in self.required_fields:
                if field not in features or features[field] is None:
                    report['missing_fields'].append(field)
            
            # Find invalid fields
            for field, value in features.items():
                if value is not None:
                    if isinstance(value, (int, float, np.number)):
                        if np.isnan(value) or np.isinf(value):
                            report['invalid_fields'].append(field)
            
            # Overall validity
            report['valid'] = (report['essential_fields_valid'] and 
                             report['optional_fields_valid'] and 
                             report['quality_valid'])
            
        except Exception as e:
            logger.error(f"Error generating validation report: {e}")
            report['error'] = str(e)
        
        return report
    
    def filter_valid_features(self, features_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter a list of feature sets to only include valid ones.
        
        Args:
            features_list (List[Dict[str, Any]]): List of feature dictionaries.
            
        Returns:
            List[Dict[str, Any]]: List of valid feature dictionaries.
        """
        valid_features = []
        invalid_count = 0
        
        for features in features_list:
            if self.is_valid_feature_set(features):
                valid_features.append(features)
            else:
                invalid_count += 1
        
        if invalid_count > 0:
            logger.info(f"Filtered out {invalid_count} invalid feature sets")
        
        return valid_features 