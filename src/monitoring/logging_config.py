#!/usr/bin/env python3
"""
Production-grade logging system with structured logging, log aggregation,
and distributed tracing support
"""

import logging
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import torch.distributed as dist
import traceback


class StructuredLogger:
    """
    Production logging with structured JSON output for log aggregation systems
    (ELK stack, Splunk, CloudWatch, etc.)
    """
    
    def __init__(
        self,
        name: str = "distributed_training",
        log_dir: str = "./logs",
        log_level: str = "INFO",
        enable_json: bool = True,
    ):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.enable_json = enable_json
        
        # Get distributed info
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        # Setup logger
        self.logger = logging.getLogger(f"{name}_rank_{self.rank}")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Console handler with custom formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        if self.enable_json:
            console_handler.setFormatter(JSONFormatter())
        else:
            console_format = logging.Formatter(
                f'%(asctime)s | Rank {self.rank}/{self.world_size} | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_format)
        
        self.logger.addHandler(console_handler)
        
        # File handler for persistent logs
        log_file = self.log_dir / f"training_rank_{self.rank}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        if self.enable_json:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(console_format)
        
        self.logger.addHandler(file_handler)
        
        # Error file for critical issues
        error_file = self.log_dir / f"errors_rank_{self.rank}.log"
        error_handler = logging.FileHandler(error_file)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(error_handler)
    
    def log_event(
        self,
        level: str,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
    ):
        """Log structured event with metadata"""
        log_data = {
            "message": message,
            "rank": self.rank,
            "world_size": self.world_size,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        if extra:
            log_data.update(extra)
        
        getattr(self.logger, level.lower())(json.dumps(log_data) if self.enable_json else message)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.log_event("INFO", message, kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.log_event("WARNING", message, kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.log_event("ERROR", message, kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.log_event("DEBUG", message, kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.log_event("CRITICAL", message, kwargs)
    
    def log_exception(self, exc: Exception, context: str = ""):
        """Log exception with full traceback"""
        self.error(
            f"Exception in {context}: {str(exc)}",
            exception_type=type(exc).__name__,
            traceback=traceback.format_exc(),
        )
    
    def log_metric(self, metric_name: str, value: float, step: int, **kwargs):
        """Log training metric"""
        self.info(
            f"Metric: {metric_name}",
            metric=metric_name,
            value=value,
            step=step,
            **kwargs
        )
    
    def log_checkpoint(self, epoch: int, checkpoint_path: str, metrics: Dict[str, float]):
        """Log checkpoint save event"""
        self.info(
            f"Checkpoint saved at epoch {epoch}",
            event_type="checkpoint_save",
            epoch=epoch,
            checkpoint_path=checkpoint_path,
            metrics=metrics,
        )
    
    def log_config(self, config: Dict[str, Any]):
        """Log training configuration"""
        self.info(
            "Training configuration",
            event_type="config",
            config=config,
        )


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for structured logging systems"""
    
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


class PerformanceTracer:
    """
    Trace performance bottlenecks with timing information
    """
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.timers = {}
    
    def start(self, name: str):
        """Start timing an operation"""
        import time
        self.timers[name] = time.time()
    
    def end(self, name: str):
        """End timing and log duration"""
        import time
        if name not in self.timers:
            self.logger.warning(f"Timer '{name}' was not started")
            return
        
        duration = time.time() - self.timers[name]
        self.logger.log_metric(
            f"timing_{name}",
            duration * 1000,  # Convert to ms
            step=0,
            event_type="performance_trace"
        )
        del self.timers[name]
        return duration


class AlertManager:
    """
    Alert system for critical events (integrates with PagerDuty, Slack, etc.)
    """
    
    def __init__(self, logger: StructuredLogger, webhook_url: Optional[str] = None):
        self.logger = logger
        self.webhook_url = webhook_url
    
    def send_alert(self, severity: str, message: str, details: Dict[str, Any] = None):
        """Send alert for critical events"""
        alert_data = {
            "severity": severity,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "rank": self.logger.rank,
        }
        
        if details:
            alert_data.update(details)
        
        # Log locally
        if severity == "CRITICAL":
            self.logger.critical(message, **alert_data)
        else:
            self.logger.error(message, **alert_data)
        
        # Send to webhook if configured
        if self.webhook_url:
            self._send_webhook(alert_data)
    
    def _send_webhook(self, data: Dict[str, Any]):
        """Send webhook notification"""
        try:
            import requests
            requests.post(
                self.webhook_url,
                json=data,
                timeout=5
            )
        except Exception as e:
            self.logger.error(f"Failed to send webhook: {e}")
    
    def alert_oom(self, gpu_id: int, memory_used: float):
        """Alert on out-of-memory error"""
        self.send_alert(
            "CRITICAL",
            f"OOM Error on GPU {gpu_id}",
            {
                "gpu_id": gpu_id,
                "memory_used_gb": memory_used,
                "event_type": "oom_error"
            }
        )
    
    def alert_training_failure(self, epoch: int, error: str):
        """Alert on training failure"""
        self.send_alert(
            "CRITICAL",
            f"Training failed at epoch {epoch}",
            {
                "epoch": epoch,
                "error": error,
                "event_type": "training_failure"
            }
        )
    
    def alert_slow_iteration(self, iteration_time: float, threshold: float = 1.0):
        """Alert if iteration is too slow"""
        if iteration_time > threshold:
            self.send_alert(
                "WARNING",
                f"Slow iteration detected: {iteration_time:.2f}s",
                {
                    "iteration_time": iteration_time,
                    "threshold": threshold,
                    "event_type": "performance_degradation"
                }
            )


# Global logger instance
_global_logger = None

def get_logger(name: str = "distributed_training", **kwargs) -> StructuredLogger:
    """Get or create global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = StructuredLogger(name, **kwargs)
    return _global_logger


if __name__ == "__main__":
    # Example usage
    logger = get_logger()
    tracer = PerformanceTracer(logger)
    alert_manager = AlertManager(logger)
    
    # Log configuration
    logger.log_config({
        "batch_size": 32,
        "learning_rate": 0.001,
        "strategy": "ddp"
    })
    
    # Log metrics
    logger.log_metric("loss", 0.5, step=100)
    
    # Trace performance
    tracer.start("forward_pass")
    # ... do work ...
    tracer.end("forward_pass")
    
    # Send alert
    alert_manager.alert_oom(gpu_id=0, memory_used=15.5)
