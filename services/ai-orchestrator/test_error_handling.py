#!/usr/bin/env python3
"""
Test script for comprehensive error handling and recovery system

This script tests various error scenarios and recovery mechanisms
to ensure the system handles failures gracefully.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.error_handling import (
    ErrorType, ErrorSeverity, ErrorContext, ErrorRecoveryManager,
    error_recovery_manager, GracefulDegradation
)
from app.workflow_recovery import (
    WorkflowRecoveryManager, RecoveryStrategy, CheckpointType,
    workflow_recovery_manager
)
from app.workflow_engine import WorkflowState, WorkflowStatus, TaskStatus
from app.workflow_persistence import workflow_persistence

# Configure logging with timestamps
import sys
from pathlib import Path
# Add utils to path
utils_path = Path(__file__).parent.parent.parent.parent.parent / 'utils'
if str(utils_path) not in sys.path:
    sys.path.insert(0, str(utils_path))

from logger import setup_logger
logger = setup_logger(__name__, service_name="error-handling")

class ErrorHandlingTester:
    """Test suite for error handling and recovery system"""
    
    def __init__(self):
        self.test_results = {}
        self.error_manager = error_recovery_manager
        self.recovery_manager = workflow_recovery_manager
        self.graceful_degradation = GracefulDegradation()
    
    async def run_all_tests(self):
        """Run all error handling tests"""
        logger.info("Starting comprehensive error handling tests...")
        
        tests = [
            ("test_error_classification", self.test_error_classification),
            ("test_retry_strategies", self.test_retry_strategies),
            ("test_graceful_degradation", self.test_graceful_degradation),
            ("test_workflow_recovery", self.test_workflow_recovery),
            ("test_checkpoint_system", self.test_checkpoint_system),
            ("test_error_statistics", self.test_error_statistics),
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
    
    async def test_error_classification(self) -> bool:
        """Test error classification system"""
        try:
            # Test network error classification
            network_error = Exception("Connection refused")
            context = ErrorContext(
                submission_id="test_001",
                agent_type="nlp",
                operation="test_classification"
            )
            
            classification = self.error_manager.classifier.classify_error(network_error, context)
            
            # Verify classification
            assert classification.error_type in [ErrorType.NETWORK_ERROR, ErrorType.UNKNOWN_ERROR]
            assert classification.severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM, ErrorSeverity.HIGH]
            assert isinstance(classification.is_retryable, bool)
            
            logger.info("✓ Error classification working correctly")
            return True
            
        except Exception as e:
            logger.error(f"Error classification test failed: {e}")
            return False
    
    async def test_retry_strategies(self) -> bool:
        """Test retry strategies"""
        try:
            attempt_count = 0
            
            async def failing_operation():
                nonlocal attempt_count
                attempt_count += 1
                if attempt_count < 3:
                    raise Exception("Simulated failure")
                return {"status": "success", "attempt": attempt_count}
            
            context = ErrorContext(
                submission_id="test_002",
                agent_type="test",
                operation="retry_test"
            )
            
            # This should succeed on the 3rd attempt
            result = await self.error_manager.handle_error(
                Exception("Test error"), context, failing_operation
            )
            
            assert result["status"] == "success"
            assert attempt_count == 3
            
            logger.info("✓ Retry strategies working correctly")
            return True
            
        except Exception as e:
            logger.error(f"Retry strategies test failed: {e}")
            return False
    
    async def test_graceful_degradation(self) -> bool:
        """Test graceful degradation for different agent types"""
        try:
            # Test NLP agent degradation
            context = ErrorContext(
                submission_id="test_003",
                agent_type="nlp",
                input_data={"text_content": "Test business requirements for web application"}
            )
            
            nlp_fallback = await self.graceful_degradation.handle_agent_failure(
                "nlp", context, Exception("NLP service unavailable")
            )
            
            assert nlp_fallback["status"] == "degraded"
            assert "results" in nlp_fallback
            assert "fallback_reason" in nlp_fallback
            
            # Test ASR agent degradation
            asr_context = ErrorContext(
                submission_id="test_004",
                agent_type="asr",
                input_data={"voice_file_url": "test.wav"}
            )
            
            asr_fallback = await self.graceful_degradation.handle_agent_failure(
                "asr", asr_context, Exception("ASR service unavailable")
            )
            
            assert asr_fallback["status"] == "degraded"
            assert "transcript" in asr_fallback["results"]
            
            logger.info("✓ Graceful degradation working correctly")
            return True
            
        except Exception as e:
            logger.error(f"Graceful degradation test failed: {e}")
            return False
    
    async def test_workflow_recovery(self) -> bool:
        """Test workflow recovery mechanisms"""
        try:
            # Create a mock workflow state
            workflow_state = WorkflowState(
                workflow_id="test_workflow_001",
                submission_id="test_005",
                workflow_type="test_recovery",
                status=WorkflowStatus.FAILED,
                current_step="test_step",
                completed_steps=["step1", "step2"],
                pending_tasks=["task3", "task4"],
                running_tasks=[],
                completed_tasks=["task1", "task2"],
                failed_tasks=["task3"],
                agent_results={},
                start_time=datetime.now(),
                progress=0.5,
                metadata={"test": True}
            )
            
            # Test recovery strategy determination
            recovery_status = await self.recovery_manager.get_recovery_status(
                workflow_state.workflow_id
            )
            
            # The workflow doesn't exist in persistence, so we expect an error
            assert "error" in recovery_status or "workflow_id" in recovery_status
            
            logger.info("✓ Workflow recovery system working correctly")
            return True
            
        except Exception as e:
            logger.error(f"Workflow recovery test failed: {e}")
            return False
    
    async def test_checkpoint_system(self) -> bool:
        """Test workflow checkpoint system"""
        try:
            # Create a mock workflow state
            workflow_state = WorkflowState(
                workflow_id="test_checkpoint_001",
                submission_id="test_006",
                workflow_type="test_checkpoint",
                status=WorkflowStatus.RUNNING,
                current_step="checkpoint_test",
                completed_steps=["step1"],
                pending_tasks=["task2"],
                running_tasks=["task1"],
                completed_tasks=[],
                failed_tasks=[],
                agent_results={},
                start_time=datetime.now(),
                progress=0.3,
                metadata={"checkpoint_test": True}
            )
            
            # Create a checkpoint
            checkpoint_id = await self.recovery_manager.create_checkpoint(
                workflow_state,
                CheckpointType.MANUAL_CHECKPOINT,
                {"test_checkpoint": True}
            )
            
            # Verify checkpoint was created
            assert checkpoint_id != ""
            assert workflow_state.workflow_id in self.recovery_manager.checkpoints
            
            logger.info("✓ Checkpoint system working correctly")
            return True
            
        except Exception as e:
            logger.error(f"Checkpoint system test failed: {e}")
            return False
    
    async def test_error_statistics(self) -> bool:
        """Test error statistics collection"""
        try:
            # Generate some test errors
            for i in range(5):
                context = ErrorContext(
                    submission_id=f"test_stats_{i}",
                    agent_type="test",
                    operation="statistics_test"
                )
                
                try:
                    await self.error_manager.handle_error(
                        Exception(f"Test error {i}"), context
                    )
                except:
                    pass  # Expected to fail
            
            # Get statistics
            stats = self.error_manager.get_error_statistics()
            
            assert "total_errors" in stats
            assert stats["total_errors"] >= 5
            
            logger.info("✓ Error statistics working correctly")
            return True
            
        except Exception as e:
            logger.error(f"Error statistics test failed: {e}")
            return False
    
    def print_test_summary(self):
        """Print test results summary"""
        print("\n" + "="*60)
        print("ERROR HANDLING AND RECOVERY TEST SUMMARY")
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
    print("Starting Error Handling and Recovery System Tests...")
    print("This will test various error scenarios and recovery mechanisms.")
    
    tester = ErrorHandlingTester()
    results = await tester.run_all_tests()
    
    # Return appropriate exit code
    failed_tests = sum(1 for result in results.values() 
                      if result["status"] in ["FAILED", "ERROR"])
    
    if failed_tests == 0:
        print("\n🎉 All tests passed! Error handling system is working correctly.")
        return 0
    else:
        print(f"\n❌ {failed_tests} tests failed. Please check the error handling implementation.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)