#!/usr/bin/env python3
"""
Diagnose Essentia TensorFlow support
"""
import sys

def check_essentia_tensorflow():
    """Check available TensorFlow predictors in Essentia"""
    try:
        import essentia.standard as es
        print("✅ Essentia imported successfully")
        
        # Check TensorFlow support
        tf_predictors = []
        
        # Check TensorFlowPredictor2D
        try:
            predictor = es.TensorFlowPredictor2D
            tf_predictors.append("TensorFlowPredictor2D")
            print("✅ TensorFlowPredictor2D available")
        except AttributeError:
            print("❌ TensorFlowPredictor2D not available")
        
        # Check TensorFlowPredictor
        try:
            predictor = es.TensorFlowPredictor
            tf_predictors.append("TensorFlowPredictor")
            print("✅ TensorFlowPredictor available")
        except AttributeError:
            print("❌ TensorFlowPredictor not available")
            
        # Check TensorFlow
        try:
            predictor = es.TensorFlow
            tf_predictors.append("TensorFlow")
            print("✅ TensorFlow available")
        except AttributeError:
            print("❌ TensorFlow not available")
            
        # Check TensorFlowPredictEffNetDiscogs
        try:
            predictor = es.TensorFlowPredictEffNetDiscogs
            tf_predictors.append("TensorFlowPredictEffNetDiscogs")
            print("✅ TensorFlowPredictEffNetDiscogs available")
        except AttributeError:
            print("❌ TensorFlowPredictEffNetDiscogs not available")
        
        print(f"\n📊 Summary: {len(tf_predictors)} TensorFlow predictors available")
        if tf_predictors:
            print(f"Available: {', '.join(tf_predictors)}")
        else:
            print("⚠️  No TensorFlow predictors available - Essentia without TensorFlow support")
            
        # Check if TensorFlow itself is available
        try:
            import tensorflow as tf
            print(f"✅ TensorFlow version: {tf.__version__}")
        except ImportError:
            print("❌ TensorFlow not available in Python environment")
            
        return len(tf_predictors) > 0
        
    except ImportError as e:
        print(f"❌ Failed to import Essentia: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Diagnosing Essentia TensorFlow Support...\n")
    has_tf_support = check_essentia_tensorflow()
    
    if has_tf_support:
        print("\n🎉 MusiCNN should work with TensorFlow models!")
    else:
        print("\n⚠️  MusiCNN will use descriptive analysis fallback")
        print("   This is still functional but won't use deep learning features")
