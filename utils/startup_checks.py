# utils/startup_checks.py
"""
Production Startup Validation
=============================
Validates all required services and configuration before starting the application.
Implements comprehensive health checks and dependency verification.
"""

import logging
import sys
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class CheckStatus(str, Enum):
    """Health check status."""
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class CheckResult:
    """Result of a health check."""
    name: str
    status: CheckStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    critical: bool = True


class StartupValidator:
    """
    Validates application startup requirements.
    
    Checks:
    - Database connectivity
    - Redis connectivity
    - Required environment variables
    - External API connectivity
    - File system permissions
    - Configuration validity
    """
    
    def __init__(self, settings):
        """
        Initialize validator with application settings.
        
        Args:
            settings: Application settings object
        """
        self.settings = settings
        self.results: List[CheckResult] = []
    
    def add_result(
        self,
        name: str,
        status: CheckStatus,
        message: str,
        details: Optional[Dict] = None,
        critical: bool = True
    ):
        """Add a check result."""
        result = CheckResult(
            name=name,
            status=status,
            message=message,
            details=details,
            critical=critical
        )
        self.results.append(result)
        
        # Log the result
        if status == CheckStatus.FAIL:
            if critical:
                logger.error(f"❌ CRITICAL: {name} - {message}")
            else:
                logger.warning(f"⚠️  WARNING: {name} - {message}")
        elif status == CheckStatus.WARN:
            logger.warning(f"⚠️  {name} - {message}")
        elif status == CheckStatus.PASS:
            logger.info(f"✅ {name} - {message}")
        else:
            logger.debug(f"⏭️  {name} - {message}")
    
    def check_database(self) -> CheckResult:
        """Check database connectivity."""
        try:
            from utils.db import test_db_connection, get_pool_status
            
            # Test connection
            if test_db_connection():
                pool_status = get_pool_status()
                self.add_result(
                    "Database Connection",
                    CheckStatus.PASS,
                    f"Connected to {pool_status.get('database', 'database')}",
                    details=pool_status,
                    critical=True
                )
            else:
                self.add_result(
                    "Database Connection",
                    CheckStatus.FAIL,
                    "Cannot connect to database",
                    critical=True
                )
        except Exception as e:
            self.add_result(
                "Database Connection",
                CheckStatus.FAIL,
                f"Database check failed: {str(e)}",
                critical=True
            )
    
    def check_redis(self) -> CheckResult:
        """Check Redis connectivity."""
        if not self.settings.REDIS_URL:
            self.add_result(
                "Redis Connection",
                CheckStatus.SKIP,
                "Redis not configured",
                critical=False
            )
            return
        
        try:
            import redis
            r = redis.from_url(
                self.settings.REDIS_URL,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            r.ping()
            
            info = r.info()
            self.add_result(
                "Redis Connection",
                CheckStatus.PASS,
                f"Connected (version: {info.get('redis_version', 'unknown')})",
                details={'version': info.get('redis_version'), 'uptime': info.get('uptime_in_seconds')},
                critical=False
            )
        except Exception as e:
            self.add_result(
                "Redis Connection",
                CheckStatus.WARN,
                f"Cannot connect to Redis: {str(e)}",
                critical=False
            )
    
    def check_environment_variables(self) -> CheckResult:
        """Check required environment variables."""
        required_vars = {
            'SECRET_KEY': True,  # Critical
            'DATABASE_PASSWORD': True,  # Critical
        }
        
        optional_vars = {
            'ANTHROPIC_API_KEY': 'AI optimization features will be limited',
            'SERPAPI_API_KEY': 'Google Trends features will be limited',
            'YOUTUBE_API_KEY': 'YouTube API features will be limited',
            'SENTRY_DSN': 'Error tracking disabled',
        }
        
        missing_critical = []
        missing_optional = []
        
        # Check critical vars
        for var, is_critical in required_vars.items():
            value = getattr(self.settings, var, None)
            if not value or value == "":
                missing_critical.append(var)
        
        if missing_critical:
            self.add_result(
                "Environment Variables",
                CheckStatus.FAIL,
                f"Missing critical variables: {', '.join(missing_critical)}",
                critical=True
            )
        else:
            # Check optional vars
            for var, description in optional_vars.items():
                value = getattr(self.settings, var, None)
                if not value:
                    missing_optional.append(f"{var} ({description})")
            
            if missing_optional:
                self.add_result(
                    "Environment Variables",
                    CheckStatus.WARN,
                    f"Optional variables not set: {len(missing_optional)} variables",
                    details={'missing': missing_optional},
                    critical=False
                )
            else:
                self.add_result(
                    "Environment Variables",
                    CheckStatus.PASS,
                    "All variables configured",
                    critical=False
                )
    
    def check_anthropic_api(self) -> CheckResult:
        """Check Anthropic API connectivity."""
        if not self.settings.ANTHROPIC_API_KEY:
            self.add_result(
                "Anthropic API",
                CheckStatus.SKIP,
                "API key not configured",
                critical=False
            )
            return
        
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.settings.ANTHROPIC_API_KEY)
            
            # Test with a minimal request
            response = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            
            if response:
                self.add_result(
                    "Anthropic API",
                    CheckStatus.PASS,
                    f"Connected successfully (model: {self.settings.ANTHROPIC_DEFAULT_MODEL})",
                    critical=False
                )
        except Exception as e:
            self.add_result(
                "Anthropic API",
                CheckStatus.WARN,
                f"API test failed: {str(e)}",
                critical=False
            )
    
    def check_serpapi(self) -> CheckResult:
        """Check SerpAPI connectivity."""
        if not self.settings.SERPAPI_API_KEY:
            self.add_result(
                "SerpAPI",
                CheckStatus.SKIP,
                "API key not configured",
                critical=False
            )
            return
        
        try:
            import requests
            response = requests.get(
                "https://serpapi.com/account.json",
                params={"api_key": self.settings.SERPAPI_API_KEY},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                self.add_result(
                    "SerpAPI",
                    CheckStatus.PASS,
                    "Connected successfully",
                    details={'searches_left': data.get('total_searches_left')},
                    critical=False
                )
            else:
                self.add_result(
                    "SerpAPI",
                    CheckStatus.WARN,
                    f"API test returned status {response.status_code}",
                    critical=False
                )
        except Exception as e:
            self.add_result(
                "SerpAPI",
                CheckStatus.WARN,
                f"API test failed: {str(e)}",
                critical=False
            )
    
    def check_file_permissions(self) -> CheckResult:
        """Check file system permissions."""
        import os
        
        # Check log directory
        log_dir = "logs"
        try:
            os.makedirs(log_dir, exist_ok=True)
            test_file = os.path.join(log_dir, ".test_write")
            
            with open(test_file, 'w') as f:
                f.write("test")
            
            os.remove(test_file)
            
            self.add_result(
                "File Permissions",
                CheckStatus.PASS,
                f"Write access to {log_dir}/ confirmed",
                critical=False
            )
        except Exception as e:
            self.add_result(
                "File Permissions",
                CheckStatus.WARN,
                f"Cannot write to {log_dir}/: {str(e)}",
                critical=False
            )
    
    def check_configuration_validity(self) -> CheckResult:
        """Validate configuration settings."""
        issues = []
        
        # Check port range
        if not (1 <= self.settings.PORT <= 65535):
            issues.append(f"Invalid port: {self.settings.PORT}")
        
        # Check pool sizes
        if self.settings.DATABASE_POOL_SIZE < 1:
            issues.append("Database pool size must be >= 1")
        
        # Check environment
        valid_envs = ['development', 'staging', 'production']
        if self.settings.ENVIRONMENT not in valid_envs:
            issues.append(f"Invalid environment: {self.settings.ENVIRONMENT}")
        
        # Check secret key length
        if len(self.settings.SECRET_KEY) < 32:
            issues.append("SECRET_KEY should be at least 32 characters")
        
        if issues:
            self.add_result(
                "Configuration Validity",
                CheckStatus.WARN,
                f"Configuration issues: {len(issues)}",
                details={'issues': issues},
                critical=False
            )
        else:
            self.add_result(
                "Configuration Validity",
                CheckStatus.PASS,
                "All configuration values valid",
                critical=False
            )
    
    def check_python_version(self) -> CheckResult:
        """Check Python version."""
        required_version = (3, 11)
        current_version = sys.version_info[:2]
        
        if current_version >= required_version:
            self.add_result(
                "Python Version",
                CheckStatus.PASS,
                f"Python {sys.version.split()[0]}",
                critical=False
            )
        else:
            self.add_result(
                "Python Version",
                CheckStatus.WARN,
                f"Python {current_version[0]}.{current_version[1]} (recommended: {required_version[0]}.{required_version[1]}+)",
                critical=False
            )
    
    def check_dependencies(self) -> CheckResult:
        """Check critical dependencies."""
        required_packages = {
            'fastapi': True,
            'psycopg2': True,
            'redis': False,
            'anthropic': False,
            'numpy': False,
        }
        
        missing_critical = []
        missing_optional = []
        
        for package, is_critical in required_packages.items():
            try:
                __import__(package)
            except ImportError:
                if is_critical:
                    missing_critical.append(package)
                else:
                    missing_optional.append(package)
        
        if missing_critical:
            self.add_result(
                "Dependencies",
                CheckStatus.FAIL,
                f"Missing critical packages: {', '.join(missing_critical)}",
                critical=True
            )
        elif missing_optional:
            self.add_result(
                "Dependencies",
                CheckStatus.WARN,
                f"Missing optional packages: {', '.join(missing_optional)}",
                critical=False
            )
        else:
            self.add_result(
                "Dependencies",
                CheckStatus.PASS,
                "All required packages installed",
                critical=False
            )
    
    def run_all_checks(self) -> bool:
        """
        Run all startup checks.
        
        Returns:
            True if all critical checks pass, False otherwise
        """
        logger.info("=" * 80)
        logger.info("Running startup validation checks...")
        logger.info("=" * 80)
        
        # Run all checks
        self.check_python_version()
        self.check_dependencies()
        self.check_environment_variables()
        self.check_configuration_validity()
        self.check_database()
        self.check_redis()
        self.check_file_permissions()
        
        # Optional API checks (non-blocking)
        if self.settings.ENABLE_LLM_OPTIMIZATION:
            self.check_anthropic_api()
        if self.settings.SERPAPI_API_KEY:
            self.check_serpapi()
        
        # Evaluate results
        critical_failures = [r for r in self.results if r.status == CheckStatus.FAIL and r.critical]
        warnings = [r for r in self.results if r.status == CheckStatus.WARN]
        passes = [r for r in self.results if r.status == CheckStatus.PASS]
        
        # Print summary
        logger.info("=" * 80)
        logger.info("Startup Validation Summary:")
        logger.info(f"  ✅ Passed: {len(passes)}")
        logger.info(f"  ⚠️  Warnings: {len(warnings)}")
        logger.info(f"  ❌ Critical Failures: {len(critical_failures)}")
        logger.info("=" * 80)
        
        if critical_failures:
            logger.error("CRITICAL FAILURES DETECTED:")
            for result in critical_failures:
                logger.error(f"  - {result.name}: {result.message}")
            logger.error("=" * 80)
            logger.error("APPLICATION CANNOT START - Fix critical issues and restart")
            return False
        
        if warnings:
            logger.warning("Warnings detected (non-critical):")
            for result in warnings:
                logger.warning(f"  - {result.name}: {result.message}")
        
        logger.info("✅ All critical startup checks passed")
        logger.info("=" * 80)
        return True
    
    def get_health_report(self) -> Dict[str, Any]:
        """
        Get a comprehensive health report.
        
        Returns:
            Dictionary with health status
        """
        return {
            'healthy': all(r.status != CheckStatus.FAIL for r in self.results if r.critical),
            'checks': [
                {
                    'name': r.name,
                    'status': r.status.value,
                    'message': r.message,
                    'critical': r.critical,
                    'details': r.details
                }
                for r in self.results
            ],
            'summary': {
                'total': len(self.results),
                'passed': len([r for r in self.results if r.status == CheckStatus.PASS]),
                'warnings': len([r for r in self.results if r.status == CheckStatus.WARN]),
                'failures': len([r for r in self.results if r.status == CheckStatus.FAIL]),
                'skipped': len([r for r in self.results if r.status == CheckStatus.SKIP]),
            }
        }


async def validate_startup(settings) -> bool:
    """
    Validate application startup.
    
    Args:
        settings: Application settings
        
    Returns:
        True if validation passes, False otherwise
    """
    validator = StartupValidator(settings)
    return validator.run_all_checks()


def get_startup_health_report(settings) -> Dict[str, Any]:
    """
    Get startup health report.
    
    Args:
        settings: Application settings
        
    Returns:
        Health report dictionary
    """
    validator = StartupValidator(settings)
    validator.run_all_checks()
    return validator.get_health_report()
