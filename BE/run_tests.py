#!/usr/bin/env python
"""
Professional Test Report Generator for Django
Generates HTML and PDF reports showing test results as proof of execution
"""
import os
import sys
import django
from django.test.utils import get_runner
from django.conf import settings
from django.test.runner import DiscoverRunner
import time
import json
from datetime import datetime
from pathlib import Path

class TestReportGenerator(DiscoverRunner):
    """Custom test runner that generates professional test reports"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = None
        self.test_results = []
        self.summary = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'skipped': 0,
            'duration': 0
        }
    
    def setup_test_environment(self, **kwargs):
        super().setup_test_environment(**kwargs)
        self.start_time = time.time()
        print("🚀 Starting Test Execution - Report Generation Enabled")
        print("=" * 70)
    
    def run_tests(self, test_labels, **kwargs):
        """Override run_tests to capture individual test results"""
        # Run the actual tests
        result = super().run_tests(test_labels, **kwargs)
        
        # Generate reports
        self.generate_reports()
        
        return result
    
    def suite_result(self, suite, result, **kwargs):
        """Capture detailed results for each test"""
        super().suite_result(suite, result, **kwargs)
        
        # Process results
        self.summary['total'] = result.testsRun
        self.summary['failed'] = len(result.failures)
        self.summary['errors'] = len(result.errors)
        self.summary['skipped'] = len(result.skipped)
        self.summary['passed'] = result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)
        self.summary['duration'] = time.time() - self.start_time if self.start_time else 0
        
        # Collect individual test results
        for test, traceback in result.failures:
            self.test_results.append({
                'name': str(test),
                'status': 'FAILED',
                'error': traceback,
                'duration': 0  # Django doesn't provide individual test timing by default
            })
        
        for test, traceback in result.errors:
            self.test_results.append({
                'name': str(test),
                'status': 'ERROR', 
                'error': traceback,
                'duration': 0
            })
        
        for test, reason in result.skipped:
            self.test_results.append({
                'name': str(test),
                'status': 'SKIPPED',
                'error': reason,
                'duration': 0
            })
        
        # Add passed tests (we need to infer these)
        all_tests = self._get_all_test_names(suite)
        failed_test_names = {str(test) for test, _ in result.failures}
        error_test_names = {str(test) for test, _ in result.errors}
        skipped_test_names = {str(test) for test, _ in result.skipped}
        
        for test_name in all_tests:
            if test_name not in failed_test_names and test_name not in error_test_names and test_name not in skipped_test_names:
                self.test_results.append({
                    'name': test_name,
                    'status': 'PASSED',
                    'error': None,
                    'duration': 0
                })
    
    def _get_all_test_names(self, suite):
        """Extract all test names from test suite"""
        test_names = []
        
        def extract_tests(test_item):
            if hasattr(test_item, '__iter__'):
                for item in test_item:
                    extract_tests(item)
            else:
                if hasattr(test_item, '_testMethodName'):
                    test_names.append(str(test_item))
        
        extract_tests(suite)
        return test_names
    
    def generate_reports(self):
        """Generate HTML and JSON test reports"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Create reports directory
        reports_dir = Path("test_reports")
        reports_dir.mkdir(exist_ok=True)
        
        # Generate JSON report
        json_report = self.generate_json_report()
        json_file = reports_dir / f"test_report_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(json_report, f, indent=2, default=str)
        
        # Generate HTML report
        html_report = self.generate_html_report()
        html_file = reports_dir / f"test_report_{timestamp}.html"
        with open(html_file, 'w') as f:
            f.write(html_report)
        
        # Generate summary text report
        text_report = self.generate_text_report()
        text_file = reports_dir / f"test_summary_{timestamp}.txt"
        with open(text_file, 'w') as f:
            f.write(text_report)
        
        print(f"\n📊 Test Reports Generated:")
        print(f"   • HTML Report: {html_file.absolute()}")
        print(f"   • JSON Report: {json_file.absolute()}")
        print(f"   • Text Summary: {text_file.absolute()}")
        print(f"\n✅ Open {html_file.absolute()} in your browser to view the detailed report")
    
    def generate_json_report(self):
        """Generate JSON format report"""
        return {
            "test_execution": {
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": round(self.summary['duration'], 2),
                "environment": {
                    "python_version": sys.version,
                    "django_version": django.get_version(),
                    "platform": sys.platform
                }
            },
            "summary": self.summary,
            "test_results": self.test_results
        }
    
    def generate_text_report(self):
        """Generate plain text summary report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        duration = round(self.summary['duration'], 2)
        
        report = f"""
UNIT TEST EXECUTION REPORT
==========================

Execution Details:
• Date & Time: {timestamp}
• Duration: {duration} seconds
• Python Version: {sys.version.split()[0]}
• Django Version: {django.get_version()}

Test Summary:
• Total Tests: {self.summary['total']}
• Passed: {self.summary['passed']} ✓
• Failed: {self.summary['failed']} ✗
• Errors: {self.summary['errors']} ⚠
• Skipped: {self.summary['skipped']} ⏭

Success Rate: {(self.summary['passed'] / max(self.summary['total'], 1)) * 100:.1f}%

Overall Result: {'PASSED' if self.summary['failed'] == 0 and self.summary['errors'] == 0 else 'FAILED'}

Detailed Test Results:
----------------------
"""
        
        for test in sorted(self.test_results, key=lambda x: x['name']):
            status_icon = {'PASSED': '✅', 'FAILED': '❌', 'ERROR': '⚠️', 'SKIPPED': '⏭️'}.get(test['status'], '❓')
            report += f"{status_icon} {test['name']} - {test['status']}\n"
            if test['error'] and test['status'] in ['FAILED', 'ERROR']:
                # Show first line of error for brevity in text report
                first_error_line = test['error'].split('\n')[0] if test['error'] else 'No details'
                report += f"    └─ {first_error_line}\n"
        
        report += f"\n{'='*50}\nReport generated on {timestamp}\n"
        return report
    
    def generate_html_report(self):
        """Generate professional HTML report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        duration = round(self.summary['duration'], 2)
        success_rate = (self.summary['passed'] / max(self.summary['total'], 1)) * 100
        
        # Status colors and icons
        status_config = {
            'PASSED': {'color': '#4CAF50', 'bg': '#E8F5E8', 'icon': '✅'},
            'FAILED': {'color': '#f44336', 'bg': '#FFEBEE', 'icon': '❌'},
            'ERROR': {'color': '#FF9800', 'bg': '#FFF3E0', 'icon': '⚠️'},
            'SKIPPED': {'color': '#9E9E9E', 'bg': '#F5F5F5', 'icon': '⏭️'}
        }
        
        # Generate test rows
        test_rows = ""
        for i, test in enumerate(sorted(self.test_results, key=lambda x: x['name']), 1):
            status = test['status']
            config = status_config.get(status, {'color': '#000', 'bg': '#FFF', 'icon': '❓'})
            
            error_detail = ""
            if test['error'] and status in ['FAILED', 'ERROR']:
                error_detail = f"""
                <tr>
                    <td colspan="4" style="padding: 10px; background: #f9f9f9; border-left: 4px solid {config['color']};">
                        <details>
                            <summary style="cursor: pointer; font-weight: bold;">View Error Details</summary>
                            <pre style="margin-top: 10px; font-size: 12px; overflow-x: auto; background: white; padding: 10px; border: 1px solid #ddd;">{test['error']}</pre>
                        </details>
                    </td>
                </tr>
                """
            
            test_rows += f"""
            <tr>
                <td style="padding: 8px; text-align: center;">{i}</td>
                <td style="padding: 8px;">{test['name']}</td>
                <td style="padding: 8px; text-align: center;">
                    <span style="background: {config['bg']}; color: {config['color']}; padding: 4px 8px; border-radius: 4px; font-weight: bold;">
                        {config['icon']} {status}
                    </span>
                </td>
                <td style="padding: 8px; text-align: center;">{test['duration']}s</td>
            </tr>
            {error_detail}
            """
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Unit Test Execution Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; padding-bottom: 20px; border-bottom: 3px solid #2196F3; }}
        .header h1 {{ color: #1976D2; margin: 0; font-size: 2.5em; }}
        .header p {{ color: #666; font-size: 1.1em; margin: 10px 0 0 0; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }}
        .summary-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }}
        .summary-card h3 {{ margin: 0 0 10px 0; font-size: 2.5em; }}
        .summary-card p {{ margin: 0; opacity: 0.9; }}
        .results-section {{ margin-top: 40px; }}
        .results-section h2 {{ color: #1976D2; border-bottom: 2px solid #2196F3; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background: white; }}
        th {{ background: linear-gradient(135deg, #2196F3, #1976D2); color: white; padding: 12px; text-align: left; }}
        td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        .footer {{ margin-top: 40px; text-align: center; color: #666; border-top: 1px solid #ddd; padding-top: 20px; }}
        .status-badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-weight: bold; }}
        .overall-result {{ font-size: 1.5em; font-weight: bold; text-align: center; padding: 20px; border-radius: 10px; margin: 20px 0; }}
        .result-pass {{ background: #E8F5E8; color: #4CAF50; }}
        .result-fail {{ background: #FFEBEE; color: #f44336; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 Unit Test Execution Report</h1>
            <p>Generated on {timestamp}</p>
        </div>
        
        <div class="overall-result {'result-pass' if self.summary['failed'] == 0 and self.summary['errors'] == 0 else 'result-fail'}">
            {'🎉 ALL TESTS PASSED' if self.summary['failed'] == 0 and self.summary['errors'] == 0 else '❌ SOME TESTS FAILED'}
        </div>
        
        <div class="summary">
            <div class="summary-card">
                <h3>{self.summary['total']}</h3>
                <p>Total Tests</p>
            </div>
            <div class="summary-card" style="background: linear-gradient(135deg, #4CAF50, #45a049);">
                <h3>{self.summary['passed']}</h3>
                <p>Passed</p>
            </div>
            <div class="summary-card" style="background: linear-gradient(135deg, #f44336, #d32f2f);">
                <h3>{self.summary['failed']}</h3>
                <p>Failed</p>
            </div>
            <div class="summary-card" style="background: linear-gradient(135deg, #FF9800, #f57c00);">
                <h3>{self.summary['errors']}</h3>
                <p>Errors</p>
            </div>
            <div class="summary-card" style="background: linear-gradient(135deg, #9E9E9E, #757575);">
                <h3>{self.summary['skipped']}</h3>
                <p>Skipped</p>
            </div>
            <div class="summary-card" style="background: linear-gradient(135deg, #9C27B0, #7B1FA2);">
                <h3>{success_rate:.1f}%</h3>
                <p>Success Rate</p>
            </div>
        </div>
        
        <div class="results-section">
            <h2>📋 Detailed Test Results</h2>
            <p><strong>Execution Time:</strong> {duration} seconds</p>
            <p><strong>Environment:</strong> Python {sys.version.split()[0]}, Django {django.get_version()}</p>
            
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Test Name</th>
                        <th>Status</th>
                        <th>Duration</th>
                    </tr>
                </thead>
                <tbody>
                    {test_rows}
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>Report generated by Django Test Report Generator</p>
            <p>This document serves as proof of unit test execution</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html_template

if __name__ == "__main__":
    # Add project to path
    project_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_dir)
    
    # Set Django settings
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BE.settings")
    
    # Setup Django
    django.setup()
    
    # Create custom test runner
    test_runner = TestReportGenerator(verbosity=2, interactive=True, keepdb=True)
    
    print("🔧 Running tests with report generation...")
    
    # Run tests
    failures = test_runner.run_tests([
        "slotifyBE.tests.test_views"
    ])
    
    sys.exit(bool(failures))
