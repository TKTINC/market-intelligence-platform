#!/usr/bin/env python3
"""
=============================================================================
SECURITY SCORE CALCULATION SCRIPT
Analyzes security scan results and calculates overall security score
=============================================================================
"""

import json
import sys
import os
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SecurityScore:
    bandit_score: float
    safety_score: float
    semgrep_score: float
    npm_audit_score: float
    overall_score: float
    grade: str
    recommendations: List[str]
    detailed_findings: Dict[str, Any]

def parse_bandit_results(bandit_file: str) -> tuple[float, Dict[str, Any]]:
    """Parse Bandit security scan results."""
    try:
        if not os.path.exists(bandit_file):
            logger.warning(f"Bandit file not found: {bandit_file}")
            return 50.0, {"error": "File not found"}
        
        with open(bandit_file, 'r') as f:
            data = json.load(f)
        
        results = data.get('results', [])
        
        # Count issues by severity
        high_issues = len([r for r in results if r.get('issue_severity') == 'HIGH'])
        medium_issues = len([r for r in results if r.get('issue_severity') == 'MEDIUM'])
        low_issues = len([r for r in results if r.get('issue_severity') == 'LOW'])
        
        # Calculate score (start at 100, deduct points)
        score = 100.0
        score -= high_issues * 20  # High severity: -20 points each
        score -= medium_issues * 10  # Medium severity: -10 points each
        score -= low_issues * 2  # Low severity: -2 points each
        
        score = max(0.0, score)  # Ensure non-negative
        
        findings = {
            'total_issues': len(results),
            'high_severity': high_issues,
            'medium_severity': medium_issues,
            'low_severity': low_issues,
            'files_scanned': data.get('metrics', {}).get('_totals', {}).get('loc', 0)
        }
        
        logger.info(f"Bandit analysis: {high_issues} high, {medium_issues} medium, {low_issues} low severity issues")
        return score, findings
        
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logger.error(f"Error parsing Bandit results: {e}")
        return 0.0, {"error": str(e)}

def parse_safety_results(safety_file: str) -> tuple[float, Dict[str, Any]]:
    """Parse Safety dependency scan results."""
    try:
        if not os.path.exists(safety_file):
            logger.warning(f"Safety file not found: {safety_file}")
            return 70.0, {"error": "File not found"}
        
        with open(safety_file, 'r') as f:
            data = json.load(f)
        
        vulnerabilities = data.get('vulnerabilities', [])
        
        # Count vulnerabilities by severity
        critical_vulns = len([v for v in vulnerabilities if v.get('severity', '').lower() == 'critical'])
        high_vulns = len([v for v in vulnerabilities if v.get('severity', '').lower() == 'high'])
        medium_vulns = len([v for v in vulnerabilities if v.get('severity', '').lower() == 'medium'])
        low_vulns = len([v for v in vulnerabilities if v.get('severity', '').lower() == 'low'])
        
        # Calculate score
        score = 100.0
        score -= critical_vulns * 25  # Critical: -25 points each
        score -= high_vulns * 15  # High: -15 points each
        score -= medium_vulns * 8  # Medium: -8 points each
        score -= low_vulns * 3  # Low: -3 points each
        
        score = max(0.0, score)
        
        findings = {
            'total_vulnerabilities': len(vulnerabilities),
            'critical_severity': critical_vulns,
            'high_severity': high_vulns,
            'medium_severity': medium_vulns,
            'low_severity': low_vulns,
            'affected_packages': len(set(v.get('package_name', 'unknown') for v in vulnerabilities))
        }
        
        logger.info(f"Safety analysis: {len(vulnerabilities)} vulnerabilities found")
        return score, findings
        
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logger.error(f"Error parsing Safety results: {e}")
        return 0.0, {"error": str(e)}

def parse_semgrep_results(semgrep_file: str) -> tuple[float, Dict[str, Any]]:
    """Parse Semgrep SAST scan results."""
    try:
        if not os.path.exists(semgrep_file):
            logger.warning(f"Semgrep file not found: {semgrep_file}")
            return 60.0, {"error": "File not found"}
        
        with open(semgrep_file, 'r') as f:
            data = json.load(f)
        
        results = data.get('results', [])
        
        # Count findings by severity
        error_findings = len([r for r in results if r.get('extra', {}).get('severity') == 'ERROR'])
        warning_findings = len([r for r in results if r.get('extra', {}).get('severity') == 'WARNING'])
        info_findings = len([r for r in results if r.get('extra', {}).get('severity') == 'INFO'])
        
        # Calculate score
        score = 100.0
        score -= error_findings * 15  # Errors: -15 points each
        score -= warning_findings * 8  # Warnings: -8 points each
        score -= info_findings * 2  # Info: -2 points each
        
        score = max(0.0, score)
        
        findings = {
            'total_findings': len(results),
            'error_severity': error_findings,
            'warning_severity': warning_findings,
            'info_severity': info_findings,
            'rules_matched': len(set(r.get('check_id', 'unknown') for r in results))
        }
        
        logger.info(f"Semgrep analysis: {len(results)} total findings")
        return score, findings
        
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logger.error(f"Error parsing Semgrep results: {e}")
        return 0.0, {"error": str(e)}

def parse_npm_audit_results(npm_file: str) -> tuple[float, Dict[str, Any]]:
    """Parse NPM audit results."""
    try:
        if not os.path.exists(npm_file):
            logger.warning(f"NPM audit file not found: {npm_file}")
            return 80.0, {"error": "File not found"}
        
        with open(npm_file, 'r') as f:
            data = json.load(f)
        
        # Handle different NPM audit formats
        if 'vulnerabilities' in data:
            # npm audit --json format
            vulns = data.get('vulnerabilities', {})
            critical = sum(1 for v in vulns.values() if 'critical' in str(v.get('severity', '')).lower())
            high = sum(1 for v in vulns.values() if 'high' in str(v.get('severity', '')).lower())
            moderate = sum(1 for v in vulns.values() if 'moderate' in str(v.get('severity', '')).lower())
            low = sum(1 for v in vulns.values() if 'low' in str(v.get('severity', '')).lower())
        else:
            # Legacy format
            audit_data = data.get('metadata', {}).get('vulnerabilities', {})
            critical = audit_data.get('critical', 0)
            high = audit_data.get('high', 0)
            moderate = audit_data.get('moderate', 0)
            low = audit_data.get('low', 0)
        
        # Calculate score
        score = 100.0
        score -= critical * 20  # Critical: -20 points each
        score -= high * 12  # High: -12 points each
        score -= moderate * 6  # Moderate: -6 points each
        score -= low * 2  # Low: -2 points each
        
        score = max(0.0, score)
        
        findings = {
            'total_vulnerabilities': critical + high + moderate + low,
            'critical_severity': critical,
            'high_severity': high,
            'moderate_severity': moderate,
            'low_severity': low
        }
        
        logger.info(f"NPM audit analysis: {critical + high + moderate + low} vulnerabilities found")
        return score, findings
        
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logger.error(f"Error parsing NPM audit results: {e}")
        return 0.0, {"error": str(e)}

def calculate_overall_score(bandit_score: float, safety_score: float, 
                          semgrep_score: float, npm_score: float) -> float:
    """Calculate weighted overall security score."""
    # Weighted average based on importance
    weights = {
        'bandit': 0.3,    # Python security issues
        'safety': 0.3,    # Dependency vulnerabilities  
        'semgrep': 0.25,  # SAST findings
        'npm': 0.15       # JavaScript dependencies
    }
    
    overall = (
        bandit_score * weights['bandit'] +
        safety_score * weights['safety'] +
        semgrep_score * weights['semgrep'] +
        npm_score * weights['npm']
    )
    
    return round(overall, 2)

def determine_grade(score: float) -> str:
    """Determine letter grade based on score."""
    if score >= 95:
        return "A+"
    elif score >= 90:
        return "A"
    elif score >= 85:
        return "B+"
    elif score >= 80:
        return "B"
    elif score >= 75:
        return "C+"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"

def generate_recommendations(bandit_score: float, safety_score: float,
                           semgrep_score: float, npm_score: float,
                           overall_score: float, detailed_findings: Dict[str, Any]) -> List[str]:
    """Generate security improvement recommendations."""
    recommendations = []
    
    # Score-based recommendations
    if bandit_score < 80:
        recommendations.append("Review and fix Python security issues identified by Bandit")
    
    if safety_score < 80:
        recommendations.append("Update Python dependencies with known vulnerabilities")
    
    if semgrep_score < 80:
        recommendations.append("Address SAST findings from Semgrep analysis")
    
    if npm_score < 80:
        recommendations.append("Update Node.js dependencies with security vulnerabilities")
    
    if overall_score < 85:
        recommendations.append("Schedule comprehensive security review with development team")
    
    if overall_score < 70:
        recommendations.append("Consider implementing additional security measures and training")
    
    # Specific findings-based recommendations
    bandit_findings = detailed_findings.get('bandit', {})
    if bandit_findings.get('high_severity', 0) > 0:
        recommendations.append("Priority: Fix high-severity Bandit issues immediately")
    
    safety_findings = detailed_findings.get('safety', {})
    if safety_findings.get('critical_severity', 0) > 0:
        recommendations.append("Critical: Update packages with critical vulnerabilities")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_recommendations = []
    for rec in recommendations:
        if rec not in seen:
            unique_recommendations.append(rec)
            seen.add(rec)
    
    return unique_recommendations

def main():
    parser = argparse.ArgumentParser(description='Calculate security score from scan results')
    parser.add_argument('bandit_file', help='Path to Bandit JSON results file')
    parser.add_argument('safety_file', help='Path to Safety JSON results file')
    parser.add_argument('semgrep_file', help='Path to Semgrep JSON results file')
    parser.add_argument('--npm-file', help='Path to NPM audit JSON results file')
    parser.add_argument('--output', '-o', help='Output file for results (default: stdout)')
    parser.add_argument('--threshold', '-t', type=float, default=85.0,
                       help='Minimum score threshold for passing (default: 85.0)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Parse all scan results
        logger.info("Parsing security scan results...")
        
        bandit_score, bandit_findings = parse_bandit_results(args.bandit_file)
        safety_score, safety_findings = parse_safety_results(args.safety_file)
        semgrep_score, semgrep_findings = parse_semgrep_results(args.semgrep_file)
        
        # NPM audit is optional
        if args.npm_file:
            npm_score, npm_findings = parse_npm_audit_results(args.npm_file)
        else:
            npm_score, npm_findings = 85.0, {"note": "NPM audit not provided"}
        
        # Calculate overall score
        overall_score = calculate_overall_score(bandit_score, safety_score, semgrep_score, npm_score)
        grade = determine_grade(overall_score)
        
        # Collect detailed findings
        detailed_findings = {
            'bandit': bandit_findings,
            'safety': safety_findings,
            'semgrep': semgrep_findings,
            'npm_audit': npm_findings
        }
        
        # Generate recommendations
        recommendations = generate_recommendations(
            bandit_score, safety_score, semgrep_score, npm_score, overall_score, detailed_findings
        )
        
        # Create final security score object
        security_score = SecurityScore(
            bandit_score=bandit_score,
            safety_score=safety_score,
            semgrep_score=semgrep_score,
            npm_audit_score=npm_score,
            overall_score=overall_score,
            grade=grade,
            recommendations=recommendations,
            detailed_findings=detailed_findings
        )
        
        # Convert to JSON
        result = asdict(security_score)
        result['passed_threshold'] = overall_score >= args.threshold
        result['threshold'] = args.threshold
        result['scan_timestamp'] = os.environ.get('GITHUB_RUN_ID', 'local')
        
        # Output results
        json_output = json.dumps(result, indent=2)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(json_output)
            logger.info(f"Results written to {args.output}")
        else:
            print(json_output)
        
        # Log summary
        logger.info(f"Security Score: {overall_score}/100 (Grade: {grade})")
        logger.info(f"Threshold: {args.threshold} - {'PASSED' if result['passed_threshold'] else 'FAILED'}")
        
        if not result['passed_threshold']:
            logger.error("Security score below threshold!")
            sys.exit(1)
        
        logger.info("Security analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error calculating security score: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
