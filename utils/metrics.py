# utils/metrics.py
"""
Metrics collection and monitoring utilities.

Provides structured metrics collection for monitoring application performance
and business metrics. Supports multiple backends (Prometheus, StatsD, CloudWatch).
"""

from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime
import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class MetricsBackend(str, Enum):
    """Supported metrics backends."""
    PROMETHEUS = "prometheus"
    STATSD = "statsd"
    CLOUDWATCH = "cloudwatch"
    DATADOG = "datadog"
    CONSOLE = "console"  # For development/testing


class MetricsCollector:
    """
    Unified metrics collector supporting multiple backends.
    
    Collects application and business metrics for monitoring,
    alerting, and analytics.
    """
    
    def __init__(
        self,
        backend: MetricsBackend = MetricsBackend.CONSOLE,
        namespace: str = "channel_optimizer",
        enabled: bool = True
    ):
        """
        Initialize metrics collector.
        
        Args:
            backend: Metrics backend to use
            namespace: Namespace/prefix for all metrics
            enabled: Whether metrics collection is enabled
        """
        self.backend = backend
        self.namespace = namespace
        self.enabled = enabled
        self._client = None
        
        if self.enabled:
            self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the metrics backend client."""
        try:
            if self.backend == MetricsBackend.PROMETHEUS:
                from prometheus_client import Counter, Histogram, Gauge
                self._client = {
                    "counter": {},
                    "histogram": {},
                    "gauge": {}
                }
                logger.info("Metrics initialized with Prometheus backend")
                
            elif self.backend == MetricsBackend.STATSD:
                from statsd import StatsClient
                self._client = StatsClient(prefix=self.namespace)
                logger.info("Metrics initialized with StatsD backend")
                
            elif self.backend == MetricsBackend.CLOUDWATCH:
                import boto3
                self._client = boto3.client('cloudwatch')
                logger.info("Metrics initialized with CloudWatch backend")
                
            elif self.backend == MetricsBackend.DATADOG:
                from datadog import initialize, statsd
                initialize()
                self._client = statsd
                logger.info("Metrics initialized with Datadog backend")
                
            else:  # CONSOLE or fallback
                self._client = None
                logger.info("Metrics initialized with console backend")
                
        except ImportError as e:
            logger.warning(
                f"Failed to import metrics backend {self.backend}: {e}. "
                "Falling back to console logging."
            )
            self.backend = MetricsBackend.CONSOLE
            self._client = None
    
    def _format_metric_name(self, name: str) -> str:
        """
        Format metric name with namespace.
        
        Args:
            name: Metric name
            
        Returns:
            str: Formatted metric name
        """
        return f"{self.namespace}.{name}".replace(" ", "_").lower()
    
    def increment(
        self,
        name: str,
        value: int = 1,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Increment a counter metric.
        
        Args:
            name: Metric name
            value: Amount to increment by
            tags: Optional tags/labels for the metric
        """
        if not self.enabled:
            return
        
        metric_name = self._format_metric_name(name)
        tags = tags or {}
        
        try:
            if self.backend == MetricsBackend.PROMETHEUS:
                # Prometheus implementation
                if metric_name not in self._client["counter"]:
                    from prometheus_client import Counter
                    self._client["counter"][metric_name] = Counter(
                        metric_name,
                        f"Counter for {name}",
                        labelnames=list(tags.keys()) if tags else []
                    )
                
                if tags:
                    self._client["counter"][metric_name].labels(**tags).inc(value)
                else:
                    self._client["counter"][metric_name].inc(value)
                    
            elif self.backend == MetricsBackend.STATSD:
                self._client.incr(metric_name, value)
                
            elif self.backend == MetricsBackend.DATADOG:
                self._client.increment(
                    metric_name,
                    value=value,
                    tags=[f"{k}:{v}" for k, v in tags.items()]
                )
                
            else:  # Console
                logger.info(
                    f"METRIC [COUNTER] {metric_name}: +{value}",
                    extra={"tags": tags}
                )
                
        except Exception as e:
            logger.error(f"Error recording counter metric {metric_name}: {e}")
    
    def gauge(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Set a gauge metric (current value).
        
        Args:
            name: Metric name
            value: Current value
            tags: Optional tags/labels for the metric
        """
        if not self.enabled:
            return
        
        metric_name = self._format_metric_name(name)
        tags = tags or {}
        
        try:
            if self.backend == MetricsBackend.PROMETHEUS:
                if metric_name not in self._client["gauge"]:
                    from prometheus_client import Gauge
                    self._client["gauge"][metric_name] = Gauge(
                        metric_name,
                        f"Gauge for {name}",
                        labelnames=list(tags.keys()) if tags else []
                    )
                
                if tags:
                    self._client["gauge"][metric_name].labels(**tags).set(value)
                else:
                    self._client["gauge"][metric_name].set(value)
                    
            elif self.backend == MetricsBackend.STATSD:
                self._client.gauge(metric_name, value)
                
            elif self.backend == MetricsBackend.DATADOG:
                self._client.gauge(
                    metric_name,
                    value=value,
                    tags=[f"{k}:{v}" for k, v in tags.items()]
                )
                
            else:  # Console
                logger.info(
                    f"METRIC [GAUGE] {metric_name}: {value}",
                    extra={"tags": tags}
                )
                
        except Exception as e:
            logger.error(f"Error recording gauge metric {metric_name}: {e}")
    
    def histogram(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Record a histogram/distribution metric.
        
        Used for measuring distributions of values (e.g., request duration).
        
        Args:
            name: Metric name
            value: Value to record
            tags: Optional tags/labels for the metric
        """
        if not self.enabled:
            return
        
        metric_name = self._format_metric_name(name)
        tags = tags or {}
        
        try:
            if self.backend == MetricsBackend.PROMETHEUS:
                if metric_name not in self._client["histogram"]:
                    from prometheus_client import Histogram
                    self._client["histogram"][metric_name] = Histogram(
                        metric_name,
                        f"Histogram for {name}",
                        labelnames=list(tags.keys()) if tags else []
                    )
                
                if tags:
                    self._client["histogram"][metric_name].labels(**tags).observe(value)
                else:
                    self._client["histogram"][metric_name].observe(value)
                    
            elif self.backend == MetricsBackend.STATSD:
                self._client.timing(metric_name, value)
                
            elif self.backend == MetricsBackend.DATADOG:
                self._client.histogram(
                    metric_name,
                    value=value,
                    tags=[f"{k}:{v}" for k, v in tags.items()]
                )
                
            else:  # Console
                logger.info(
                    f"METRIC [HISTOGRAM] {metric_name}: {value}",
                    extra={"tags": tags}
                )
                
        except Exception as e:
            logger.error(f"Error recording histogram metric {metric_name}: {e}")
    
    @contextmanager
    def timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """
        Context manager for timing code execution.
        
        Args:
            name: Metric name
            tags: Optional tags/labels for the metric
            
        Example:
            with metrics.timer("database.query"):
                result = db.execute(query)
        """
        start_time = time.time()
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.histogram(f"{name}.duration", duration * 1000, tags)  # Convert to ms
    
    def record_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration: float,
        user_id: Optional[str] = None
    ):
        """
        Record HTTP request metrics.
        
        Args:
            endpoint: API endpoint path
            method: HTTP method
            status_code: Response status code
            duration: Request duration in seconds
            user_id: Optional user identifier
        """
        tags = {
            "endpoint": endpoint,
            "method": method,
            "status": str(status_code),
            "status_class": f"{status_code // 100}xx"
        }
        
        if user_id:
            tags["user_id"] = user_id
        
        self.increment("http.requests.total", tags=tags)
        self.histogram("http.request.duration", duration * 1000, tags=tags)
        
        if status_code >= 400:
            self.increment("http.requests.errors", tags=tags)
    
    def record_optimization_event(
        self,
        event_type: str,
        channel_id: int,
        optimization_id: Optional[int] = None,
        duration: Optional[float] = None,
        success: bool = True
    ):
        """
        Record optimization-specific business metrics.
        
        Args:
            event_type: Type of optimization event
            channel_id: Channel identifier
            optimization_id: Optimization identifier
            duration: Event duration in seconds
            success: Whether the event succeeded
        """
        tags = {
            "event_type": event_type,
            "channel_id": str(channel_id),
            "success": str(success)
        }
        
        if optimization_id:
            tags["optimization_id"] = str(optimization_id)
        
        self.increment(f"optimization.events.{event_type}", tags=tags)
        
        if duration is not None:
            self.histogram(f"optimization.{event_type}.duration", duration * 1000, tags=tags)
        
        if not success:
            self.increment(f"optimization.{event_type}.failures", tags=tags)
