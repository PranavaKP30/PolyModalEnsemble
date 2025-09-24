#!/usr/bin/env python3
"""
Unified Error Handling System for PolyModal Ensemble Learning Pipeline

This module provides consistent error handling, logging, and rollback mechanisms
across all stages of the pipeline.
"""

import logging
import traceback
import time
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from dataclasses import dataclass
import json
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for better organization."""
    DATA_LOADING = "data_loading"
    DATA_VALIDATION = "data_validation"
    MODEL_TRAINING = "model_training"
    CONFIGURATION = "configuration"
    MEMORY = "memory"
    FILE_IO = "file_io"
    NETWORK = "network"
    UNKNOWN = "unknown"

@dataclass
class ErrorInfo:
    """Structured error information."""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    stage: str
    timestamp: str
    traceback: str
    context: Dict[str, Any]
    recoverable: bool
    rollback_actions: List[str]

class PipelineError(Exception):
    """Custom exception for pipeline errors."""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.UNKNOWN, 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, stage: str = "unknown",
                 context: Dict[str, Any] = None, recoverable: bool = True):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.stage = stage
        self.context = context or {}
        self.recoverable = recoverable
        self.timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

class ErrorHandler:
    """
    Unified error handling system with rollback capabilities.
    """
    
    def __init__(self, log_file: Optional[str] = None, enable_rollback: bool = True):
        """
        Initialize error handler.
        
        Parameters
        ----------
        log_file : str, optional
            Path to error log file
        enable_rollback : bool
            Enable rollback mechanisms
        """
        self.log_file = log_file
        self.enable_rollback = enable_rollback
        self.error_log: List[ErrorInfo] = []
        self.rollback_stack: List[Callable] = []
        self.stage_state: Dict[str, Any] = {}
        
        # Setup file logging if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.ERROR)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    def register_rollback_action(self, action: Callable, description: str = ""):
        """Register a rollback action."""
        if self.enable_rollback:
            self.rollback_stack.append((action, description))
            logger.debug(f"Registered rollback action: {description}")
    
    def save_stage_state(self, stage: str, state: Dict[str, Any]):
        """Save current stage state for potential rollback."""
        self.stage_state[stage] = {
            'state': state,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        logger.debug(f"Saved state for stage: {stage}")
    
    def handle_error(self, error: Exception, stage: str, context: Dict[str, Any] = None) -> ErrorInfo:
        """
        Handle an error with appropriate logging and rollback.
        
        Parameters
        ----------
        error : Exception
            The error that occurred
        stage : str
            Current pipeline stage
        context : dict, optional
            Additional context information
            
        Returns
        -------
        ErrorInfo
            Structured error information
        """
        # Determine error category and severity
        category, severity = self._classify_error(error)
        
        # Create error info
        error_info = ErrorInfo(
            error_id=f"{stage}_{int(time.time())}",
            category=category,
            severity=severity,
            message=str(error),
            stage=stage,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            traceback=traceback.format_exc(),
            context=context or {},
            recoverable=self._is_recoverable(error, category),
            rollback_actions=[]
        )
        
        # Log the error
        self._log_error(error_info)
        
        # Add to error log
        self.error_log.append(error_info)
        
        # Handle based on severity
        if severity == ErrorSeverity.CRITICAL:
            self._handle_critical_error(error_info)
        elif severity == ErrorSeverity.HIGH:
            self._handle_high_severity_error(error_info)
        elif severity == ErrorSeverity.MEDIUM:
            self._handle_medium_severity_error(error_info)
        else:
            self._handle_low_severity_error(error_info)
        
        return error_info
    
    def _classify_error(self, error: Exception) -> tuple[ErrorCategory, ErrorSeverity]:
        """Classify error into category and severity."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Category classification
        if isinstance(error, FileNotFoundError) or "file" in error_message:
            category = ErrorCategory.FILE_IO
        elif isinstance(error, MemoryError) or "memory" in error_message:
            category = ErrorCategory.MEMORY
        elif isinstance(error, ValueError) and ("shape" in error_message or "dimension" in error_message):
            category = ErrorCategory.DATA_VALIDATION
        elif "config" in error_message or "parameter" in error_message:
            category = ErrorCategory.CONFIGURATION
        elif "training" in error_message or "model" in error_message:
            category = ErrorCategory.MODEL_TRAINING
        else:
            category = ErrorCategory.UNKNOWN
        
        # Severity classification
        if isinstance(error, (MemoryError, SystemError)):
            severity = ErrorSeverity.CRITICAL
        elif isinstance(error, (ValueError, TypeError)) and "critical" in error_message:
            severity = ErrorSeverity.HIGH
        elif isinstance(error, (FileNotFoundError, PermissionError)):
            severity = ErrorSeverity.MEDIUM
        else:
            severity = ErrorSeverity.LOW
        
        return category, severity
    
    def _is_recoverable(self, error: Exception, category: ErrorCategory) -> bool:
        """Determine if error is recoverable."""
        if category == ErrorCategory.MEMORY:
            return False  # Memory errors are usually not recoverable
        elif category == ErrorCategory.FILE_IO and isinstance(error, FileNotFoundError):
            return False  # Missing files are not recoverable
        else:
            return True
    
    def _log_error(self, error_info: ErrorInfo):
        """Log error with appropriate level."""
        log_message = f"[{error_info.stage}] {error_info.category.value}: {error_info.message}"
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_info.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        # Log traceback for high severity errors
        if error_info.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            logger.debug(f"Traceback: {error_info.traceback}")
    
    def _handle_critical_error(self, error_info: ErrorInfo):
        """Handle critical errors with immediate rollback."""
        logger.critical(f"CRITICAL ERROR in {error_info.stage}: {error_info.message}")
        
        if self.enable_rollback:
            self._execute_rollback(error_info.stage)
        
        # Save error report
        self._save_error_report()
        
        # Re-raise as PipelineError
        raise PipelineError(
            f"Critical error in {error_info.stage}: {error_info.message}",
            category=error_info.category,
            severity=error_info.severity,
            stage=error_info.stage,
            context=error_info.context,
            recoverable=False
        )
    
    def _handle_high_severity_error(self, error_info: ErrorInfo):
        """Handle high severity errors with selective rollback."""
        logger.error(f"HIGH SEVERITY ERROR in {error_info.stage}: {error_info.message}")
        
        if self.enable_rollback and error_info.recoverable:
            self._execute_selective_rollback(error_info.stage)
        
        # Save error report
        self._save_error_report()
    
    def _handle_medium_severity_error(self, error_info: ErrorInfo):
        """Handle medium severity errors with logging."""
        logger.warning(f"MEDIUM SEVERITY ERROR in {error_info.stage}: {error_info.message}")
        
        # Continue execution but log the issue
        pass
    
    def _handle_low_severity_error(self, error_info: ErrorInfo):
        """Handle low severity errors with minimal logging."""
        logger.info(f"LOW SEVERITY ERROR in {error_info.stage}: {error_info.message}")
        
        # Continue execution normally
        pass
    
    def _execute_rollback(self, stage: str):
        """Execute full rollback for a stage."""
        logger.info(f"Executing rollback for stage: {stage}")
        
        rollback_count = 0
        for action, description in reversed(self.rollback_stack):
            try:
                action()
                rollback_count += 1
                logger.info(f"Rollback action executed: {description}")
            except Exception as e:
                logger.error(f"Rollback action failed: {description} - {e}")
        
        logger.info(f"Rollback completed: {rollback_count} actions executed")
    
    def _execute_selective_rollback(self, stage: str):
        """Execute selective rollback based on error type."""
        logger.info(f"Executing selective rollback for stage: {stage}")
        
        # Only rollback actions related to the current stage
        stage_rollbacks = [action for action, desc in self.rollback_stack if stage in desc]
        
        for action, description in reversed(stage_rollbacks):
            try:
                action()
                logger.info(f"Selective rollback executed: {description}")
            except Exception as e:
                logger.error(f"Selective rollback failed: {description} - {e}")
    
    def _save_error_report(self):
        """Save comprehensive error report."""
        if not self.log_file:
            return
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_errors': len(self.error_log),
            'errors_by_stage': {},
            'errors_by_category': {},
            'errors_by_severity': {},
            'recent_errors': []
        }
        
        # Analyze errors
        for error in self.error_log:
            stage = error.stage
            category = error.category.value
            severity = error.severity.value
            
            report['errors_by_stage'][stage] = report['errors_by_stage'].get(stage, 0) + 1
            report['errors_by_category'][category] = report['errors_by_category'].get(category, 0) + 1
            report['errors_by_severity'][severity] = report['errors_by_severity'].get(severity, 0) + 1
        
        # Get recent errors (last 10)
        recent_errors = self.error_log[-10:] if len(self.error_log) > 10 else self.error_log
        report['recent_errors'] = [
            {
                'error_id': e.error_id,
                'stage': e.stage,
                'category': e.category.value,
                'severity': e.severity.value,
                'message': e.message,
                'timestamp': e.timestamp,
                'recoverable': e.recoverable
            }
            for e in recent_errors
        ]
        
        # Save report
        report_file = self.log_file.replace('.log', '_report.json')
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Error report saved to: {report_file}")
        except Exception as e:
            logger.error(f"Failed to save error report: {e}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors."""
        if not self.error_log:
            return {'total_errors': 0, 'message': 'No errors recorded'}
        
        summary = {
            'total_errors': len(self.error_log),
            'errors_by_stage': {},
            'errors_by_category': {},
            'errors_by_severity': {},
            'recoverable_errors': 0,
            'critical_errors': 0
        }
        
        for error in self.error_log:
            stage = error.stage
            category = error.category.value
            severity = error.severity.value
            
            summary['errors_by_stage'][stage] = summary['errors_by_stage'].get(stage, 0) + 1
            summary['errors_by_category'][category] = summary['errors_by_category'].get(category, 0) + 1
            summary['errors_by_severity'][severity] = summary['errors_by_severity'].get(severity, 0) + 1
            
            if error.recoverable:
                summary['recoverable_errors'] += 1
            
            if error.severity == ErrorSeverity.CRITICAL:
                summary['critical_errors'] += 1
        
        return summary
    
    def clear_error_log(self):
        """Clear the error log."""
        self.error_log.clear()
        logger.info("Error log cleared")
    
    def is_pipeline_healthy(self) -> bool:
        """Check if pipeline is in a healthy state."""
        if not self.error_log:
            return True
        
        # Check for critical errors
        critical_errors = [e for e in self.error_log if e.severity == ErrorSeverity.CRITICAL]
        if critical_errors:
            return False
        
        # Check for too many high severity errors
        high_errors = [e for e in self.error_log if e.severity == ErrorSeverity.HIGH]
        if len(high_errors) > 5:  # Threshold for too many high severity errors
            return False
        
        return True

# Global error handler instance
global_error_handler = ErrorHandler(
    log_file="pipeline_errors.log",
    enable_rollback=True
)

def handle_pipeline_error(error: Exception, stage: str, context: Dict[str, Any] = None) -> ErrorInfo:
    """Convenience function for handling pipeline errors."""
    return global_error_handler.handle_error(error, stage, context)

def register_rollback(action: Callable, description: str = ""):
    """Convenience function for registering rollback actions."""
    global_error_handler.register_rollback_action(action, description)

def save_stage_state(stage: str, state: Dict[str, Any]):
    """Convenience function for saving stage state."""
    global_error_handler.save_stage_state(stage, state)
