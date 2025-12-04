#!/usr/bin/env python3
"""
Simple test script for error handling system

This script tests the error handling components without requiring Redis or other dependencies.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Configure logging with timestamps
import sys
from pathlib import Path
# Add utils to path
utils_path = Path(__file__).parent.parent.parent.parent.parent / 'utils'
if str(utils_path) not in sys.path:
    sys.path.insert(0, str(utils_path))

from logger import setup_logger
logger = setup_logger(__name__, service_name="error-handling-simple")

class SimpleErrorHandlingTester:
    """Simple test suite for error handling components"""
    
    def __init__(self):
        self.test_results = {}
    
    async def run_all_tests(self):
        """Run all error handling tests"""
        logger.info("Starting simple error handling tests...")
        
        tests = [
            ("test_error_types", self.test_error_types),
            ("test_error_context", self.test_error_context),
            ("test_error_classification", self.test_error_classification),
            ("test_graceful_degradation", self.test_graceful_degradation),
        ]
        
        for test_name, test_func in tests:
            try:
                logger.info(f"Running test: {test_name}")
                result = await test_func()
                self.test_results[test_name] = {
                    "status": "PASSED" if result else "FAILED",
                    "timestamp": datetime.now().isoformat()
                }
                logger.info(f"Test {test_name}: {'PASSED' if result else 'FAILED'}")
            except Exception as e:
                logger.error(f"Test {test_name} failed with exception: {e}")
                self.test_results[test_name] = {
                    "status": "ERROR",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        # Print summary
        self.print_test_summary()
        return self.test_results
    
    async def test_error_types(self) -> bool:
        """Test error type enumeration"""
        try:
            from app.error_handling import ErrorType, ErrorSeverity, RetryStrategy
            
            # Test that all error types are defined
            assert ErrorType.NETWORK_ERROR
            assert ErrorType.TIMEOUT_ERROR
            assert ErrorType.AI_SERVICE_ERROR
            assert ErrorType.VALIDATION_ERROR
            
            # Test severity levels
            assert ErrorSeverity.LOW
            assert ErrorSeverity.MEDIUM
            assert ErrorSeverity.HIGH
            assert ErrorSeverity.CRITICAL
            
            # Test retry strategies
            assert RetryStrategy.NO_RETRY
            assert RetryStrategy.LINEAR_RETRY
            assert RetryStrategy.EXPONENTIAL_BACKOFF
            
            logger.info("✓ Error types and enums defined correctly")
            return True
            
        except Exception as e:
            logger.error(f"Error types test failed: {e}")
            return False
    
    async def test_error_context(self) -> bool:
        """Test error context creation"""
        try:
            from app.error_handling import ErrorContext
            
            context = ErrorContext(
                submission_id="test_001",
                agent_type="nlp",
                operation="test_operation",
                input_data={"test": "data"}
            )
            
            assert context.submission_id == "test_001"
            assert context.agent_type == "nlp"
            assert context.operation == "test_operation"
            assert context.input_data["test"] == "data"
            assert context.attempt_number == 1
            
            logger.info("✓ Error context creation working correctly")
            return True
            
        except Exception as e:
            logger.error(f"Error context test failed: {e}")
            return False
    
    async def test_error_classification(self) -> bool:
        """Test error classification system"""
        try:
            from app.error_handling import ErrorClassifier, ErrorContext, ErrorType
            
            # Test network error classification
            network_error = Exception("Connection refused")
            context = ErrorContext(
                submission_id="test_002",
                agent_type="nlp",
                operation="test_classification"
            )
            
            classification = ErrorClassifier.classify_error(network_error, context)
            
            # Verify classification has required fields
            assert hasattr(classification, 'error_type')
            assert hasattr(classification, 'severity')
            assert hasattr(classification, 'is_retryable')
            assert hasattr(classification, 'retry_strategy')
            assert hasattr(classification, 'max_retries')
            assert hasattr(classification, 'base_delay')
            assert hasattr(classification, 'max_delay')
            
            logger.info("✓ Error classification working correctly")
            return True
            
        except Exception as e:
            logger.error(f"Error classification test failed: {e}")
            return False
    
    async def test_graceful_degradation(self) -> bool:
        """Test graceful degradation for different agent types"""
        try:
            from app.error_handling import GracefulDegradation, ErrorContext
            
            degradation = GracefulDegradation()
            
            # Test NLP agent degradation
            context = ErrorContext(
                submission_id="test_003",
                agent_type="nlp",
                input_data={"text_content": "Test business requirements for web application"}
            )
            
            nlp_fallback = await degradation.handle_agent_failure(
                "nlp", context, Exception("NLP service unavailable")
            )
            
            assert nlp_fallback["status"] == "degraded"
            assert "results" in nlp_fallback
            assert "fallback_reason" in nlp_fallback
            assert "text_summary" in nlp_fallback["results"]
            
            # Test ASR agent degradation
            asr_context = ErrorContext(
                submission_id="test_004",
                agent_type="asr",
                input_data={"voice_file_url": "test.wav"}
            )
            
            asr_fallback = await degradation.handle_agent_failure(
                "asr", asr_context, Exception("ASR service unavailable")
            )
            
            assert asr_fallback["status"] == "degraded"
            assert "transcript" in asr_fallback["results"]
            
            # Test Document agent degradation
            doc_context = ErrorContext(
                submission_id="test_005",
                agent_type="document",
                input_data={"file_urls": ["test.pdf", "test2.docx"]}
            )
            
            doc_fallback = await degradation.handle_agent_failure(
                "document", doc_context, Exception("Document AI service unavailable")
            )
            
            assert doc_fallback["status"] == "degraded"
            assert "extracted_text" in doc_fallback["results"]
            
            # Test Prototype agent degradation
            proto_context = ErrorContext(
                submission_id="test_006",
                agent_type="prototype",
                input_data={"requirements": "Build a web application"}
            )
            
            proto_fallback = await degradation.handle_agent_failure(
                "prototype", proto_context, Exception("Prototype service unavailable")
            )
            
            assert proto_fallback["status"] == "degraded"
            assert "type" in proto_fallback["results"]
            assert "tech_stack" in proto_fallback["results"]
            
            logger.info("✓ Graceful degradation working correctly for all agent types")
            return True
            
        except Exception as e:
            logger.error(f"Graceful degradation test failed: {e}")
            return False
    
    def print_test_summary(self):
        """Print test results summary"""
        print("\n" + "="*60)
        print("ERROR HANDLING SYSTEM TEST SUMMARY")
        print("="*60)
        
        passed = sum(1 for result in self.test_results.values() if result["status"] == "PASSED")
        failed = sum(1 for result in self.test_results.values() if result["status"] == "FAILED")
        errors = sum(1 for result in self.test_results.values() if result["status"] == "ERROR")
        total = len(self.test_results)
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Errors: {errors}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        print("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status_symbol = "✓" if result["status"] == "PASSED" else "✗"
            print(f"  {status_symbol} {test_name}: {result['status']}")
            if result["status"] == "ERROR":
                print(f"    Error: {result.get('error', 'Unknown error')}")
        
        print("="*60)

async def main():
    """Main test function"""
    print("Starting Simple Error Handling System Tests...")
    print("Testing core error handling components without external dependencies.")
    
    tester = SimpleErrorHandlingTester()
    results = await tester.run_all_tests()
    
    # Return appropriate exit code
    failed_tests = sum(1 for result in results.values() 
                      if result["status"] in ["FAILED", "ERROR"])
    
    if failed_tests == 0:
        print("\n🎉 All tests passed! Error handling system components are working correctly.")
        return 0
    else:
        print(f"\n❌ {failed_tests} tests failed. Please check the error handling implementation.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)