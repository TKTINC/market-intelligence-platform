#!/usr/bin/env python3
"""
Brokerage Connectivity Test Script
Tests VPC endpoints, security groups, and API connectivity for brokerage integration
"""

import asyncio
import json
import logging
import socket
import ssl
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import boto3
import requests
from botocore.exceptions import ClientError, NoCredentialsError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BrokerageConnectivityTester:
    """Test brokerage connectivity and security configurations"""
    
    def __init__(self, region: str = 'us-east-1', environment: str = 'dev'):
        self.region = region
        self.environment = environment
        self.session = boto3.Session(region_name=region)
        
        # AWS clients
        self.ec2 = self.session.client('ec2')
        self.secrets_manager = self.session.client('secretsmanager')
        self.eks = self.session.client('eks')
        self.kms = self.session.client('kms')
        
        # Test results
        self.test_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'environment': environment,
            'region': region,
            'tests': {}
        }
        
        # Brokerage endpoints
        self.brokerage_endpoints = {
            'td_ameritrade': {
                'api_url': 'https://api.tdameritrade.com',
                'auth_url': 'https://auth.tdameritrade.com',
                'ports': [443, 80]
            },
            'interactive_brokers': {
                'api_url': 'https://api.interactivebrokers.com',
                'gateway_url': 'https://localhost:5000',  # TWS Gateway
                'ports': [443, 4001, 4002, 5000]
            },
            'schwab': {
                'api_url': 'https://api.schwab.com',
                'auth_url': 'https://api.schwab.com/oauth',
                'ports': [443, 80]
            }
        }

    async def run_all_tests(self) -> Dict:
        """Run all connectivity tests"""
        logger.info("🚀 Starting brokerage connectivity tests...")
        
        tests = [
            ('vpc_endpoints', self.test_vpc_endpoints),
            ('security_groups', self.test_security_groups),
            ('secrets_manager', self.test_secrets_manager),
            ('kms_encryption', self.test_kms_encryption),
            ('network_connectivity', self.test_network_connectivity),
            ('ssl_certificates', self.test_ssl_certificates),
            ('dns_resolution', self.test_dns_resolution),
            ('api_reachability', self.test_api_reachability),
            ('oauth_endpoints', self.test_oauth_endpoints)
        ]
        
        for test_name, test_func in tests:
            logger.info(f"⏳ Running {test_name} test...")
            try:
                result = await test_func()
                self.test_results['tests'][test_name] = {
                    'status': 'PASSED' if result['success'] else 'FAILED',
                    'details': result,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                status_icon = "✅" if result['success'] else "❌"
                logger.info(f"{status_icon} {test_name}: {result.get('message', 'Completed')}")
                
            except Exception as e:
                logger.error(f"❌ {test_name} failed: {str(e)}")
                self.test_results['tests'][test_name] = {
                    'status': 'ERROR',
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }
        
        # Generate summary
        self.generate_summary()
        return self.test_results

    async def test_vpc_endpoints(self) -> Dict:
        """Test VPC endpoints for Secrets Manager and KMS"""
        try:
            # Find VPC endpoints
            response = self.ec2.describe_vpc_endpoints(
                Filters=[
                    {'Name': 'service-name', 'Values': [
                        f'com.amazonaws.{self.region}.secretsmanager',
                        f'com.amazonaws.{self.region}.kms'
                    ]},
                    {'Name': 'tag:Project', 'Values': ['MIP']}
                ]
            )
            
            endpoints = response.get('VpcEndpoints', [])
            
            if not endpoints:
                return {
                    'success': False,
                    'message': 'No VPC endpoints found for Secrets Manager or KMS',
                    'endpoints': []
                }
            
            endpoint_details = []
            for endpoint in endpoints:
                endpoint_info = {
                    'service_name': endpoint['ServiceName'],
                    'state': endpoint['State'],
                    'vpc_id': endpoint['VpcId'],
                    'private_dns_enabled': endpoint.get('PrivateDnsEnabled', False),
                    'dns_entries': [entry['DnsName'] for entry in endpoint.get('DnsEntries', [])]
                }
                endpoint_details.append(endpoint_info)
            
            all_available = all(ep['State'] == 'available' for ep in endpoints)
            
            return {
                'success': all_available,
                'message': f'Found {len(endpoints)} VPC endpoints, all available: {all_available}',
                'endpoints': endpoint_details
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'VPC endpoint test failed: {str(e)}',
                'error': str(e)
            }

    async def test_security_groups(self) -> Dict:
        """Test security group configurations for brokerage integration"""
        try:
            # Find brokerage security groups
            response = self.ec2.describe_security_groups(
                Filters=[
                    {'Name': 'tag:Purpose', 'Values': ['BrokerageIntegration']},
                    {'Name': 'tag:Project', 'Values': ['MIP']}
                ]
            )
            
            security_groups = response.get('SecurityGroups', [])
            
            if not security_groups:
                return {
                    'success': False,
                    'message': 'No brokerage security groups found'
                }
            
            sg_details = []
            for sg in security_groups:
                # Check outbound rules for HTTPS (port 443)
                https_egress = any(
                    rule.get('FromPort') == 443 and rule.get('ToPort') == 443
                    for rule in sg.get('IpPermissionsEgress', [])
                )
                
                sg_info = {
                    'group_id': sg['GroupId'],
                    'group_name': sg['GroupName'],
                    'description': sg['Description'],
                    'vpc_id': sg['VpcId'],
                    'https_egress_allowed': https_egress,
                    'ingress_rules': len(sg.get('IpPermissions', [])),
                    'egress_rules': len(sg.get('IpPermissionsEgress', []))
                }
                sg_details.append(sg_info)
            
            all_https_enabled = all(sg['https_egress_allowed'] for sg in sg_details)
            
            return {
                'success': all_https_enabled,
                'message': f'Found {len(security_groups)} security groups, HTTPS egress: {all_https_enabled}',
                'security_groups': sg_details
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Security group test failed: {str(e)}',
                'error': str(e)
            }

    async def test_secrets_manager(self) -> Dict:
        """Test Secrets Manager access and brokerage secrets"""
        try:
            secret_names = [
                f'mip-{self.environment}/brokerage/td-ameritrade',
                f'mip-{self.environment}/brokerage/interactive-brokers',
                f'mip-{self.environment}/brokerage/schwab'
            ]
            
            secret_details = []
            for secret_name in secret_names:
                try:
                    response = self.secrets_manager.describe_secret(SecretId=secret_name)
                    
                    secret_info = {
                        'name': response['Name'],
                        'arn': response['ARN'],
                        'created_date': response['CreatedDate'].isoformat(),
                        'kms_key_id': response.get('KmsKeyId'),
                        'rotation_enabled': response.get('RotationEnabled', False),
                        'last_rotated': response.get('LastRotatedDate', 'Never').isoformat() if response.get('LastRotatedDate') else 'Never',
                        'status': 'EXISTS'
                    }
                    secret_details.append(secret_info)
                    
                except ClientError as e:
                    if e.response['Error']['Code'] == 'ResourceNotFoundException':
                        secret_details.append({
                            'name': secret_name,
                            'status': 'NOT_FOUND',
                            'error': 'Secret does not exist'
                        })
                    else:
                        raise
            
            existing_secrets = [s for s in secret_details if s.get('status') == 'EXISTS']
            
            return {
                'success': len(existing_secrets) > 0,
                'message': f'Found {len(existing_secrets)}/{len(secret_names)} brokerage secrets',
                'secrets': secret_details
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Secrets Manager test failed: {str(e)}',
                'error': str(e)
            }

    async def test_kms_encryption(self) -> Dict:
        """Test KMS key access and encryption capabilities"""
        try:
            # Find trading secrets KMS key
            response = self.kms.list_aliases()
            
            trading_key_alias = f'alias/mip-{self.environment}-trading-secrets'
            trading_key = None
            
            for alias in response.get('Aliases', []):
                if alias['AliasName'] == trading_key_alias:
                    trading_key = alias
                    break
            
            if not trading_key:
                return {
                    'success': False,
                    'message': f'Trading secrets KMS key alias not found: {trading_key_alias}'
                }
            
            # Test key access
            key_response = self.kms.describe_key(KeyId=trading_key['TargetKeyId'])
            
            key_info = {
                'key_id': key_response['KeyMetadata']['KeyId'],
                'arn': key_response['KeyMetadata']['Arn'],
                'description': key_response['KeyMetadata']['Description'],
                'key_usage': key_response['KeyMetadata']['KeyUsage'],
                'key_state': key_response['KeyMetadata']['KeyState'],
                'creation_date': key_response['KeyMetadata']['CreationDate'].isoformat(),
                'key_rotation_enabled': key_response['KeyMetadata'].get('KeyRotationStatus', False)
            }
            
            # Test encryption/decryption
            test_data = b"test-brokerage-credential-encryption"
            try:
                encrypt_response = self.kms.encrypt(
                    KeyId=trading_key['TargetKeyId'],
                    Plaintext=test_data
                )
                
                decrypt_response = self.kms.decrypt(
                    CiphertextBlob=encrypt_response['CiphertextBlob']
                )
                
                encryption_test_passed = decrypt_response['Plaintext'] == test_data
                
            except Exception as e:
                encryption_test_passed = False
                key_info['encryption_test_error'] = str(e)
            
            return {
                'success': key_info['key_state'] == 'Enabled' and encryption_test_passed,
                'message': f'KMS key state: {key_info["key_state"]}, encryption test: {encryption_test_passed}',
                'key_info': key_info,
                'encryption_test_passed': encryption_test_passed
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'KMS encryption test failed: {str(e)}',
                'error': str(e)
            }

    async def test_network_connectivity(self) -> Dict:
        """Test network connectivity to brokerage endpoints"""
        connectivity_results = {}
        
        for broker, config in self.brokerage_endpoints.items():
            broker_results = {}
            
            # Test each endpoint
            for endpoint_type, url in config.items():
                if endpoint_type == 'ports':
                    continue
                    
                try:
                    # Parse URL to get hostname
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    hostname = parsed.hostname
                    port = parsed.port or 443
                    
                    # Test socket connection
                    start_time = time.time()
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(10)
                    
                    result = sock.connect_ex((hostname, port))
                    connection_time = time.time() - start_time
                    sock.close()
                    
                    broker_results[endpoint_type] = {
                        'url': url,
                        'hostname': hostname,
                        'port': port,
                        'reachable': result == 0,
                        'connection_time_ms': round(connection_time * 1000, 2),
                        'error_code': result if result != 0 else None
                    }
                    
                except Exception as e:
                    broker_results[endpoint_type] = {
                        'url': url,
                        'reachable': False,
                        'error': str(e)
                    }
            
            connectivity_results[broker] = broker_results
        
        # Check overall success
        total_tests = sum(len(broker_results) for broker_results in connectivity_results.values())
        successful_tests = sum(
            1 for broker_results in connectivity_results.values()
            for endpoint_result in broker_results.values()
            if endpoint_result.get('reachable', False)
        )
        
        return {
            'success': successful_tests > 0,
            'message': f'{successful_tests}/{total_tests} endpoint connections successful',
            'connectivity_results': connectivity_results
        }

    async def test_ssl_certificates(self) -> Dict:
        """Test SSL certificate validity for brokerage endpoints"""
        ssl_results = {}
        
        for broker, config in self.brokerage_endpoints.items():
            broker_ssl_results = {}
            
            for endpoint_type, url in config.items():
                if endpoint_type == 'ports':
                    continue
                    
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    hostname = parsed.hostname
                    port = parsed.port or 443
                    
                    # Get SSL certificate info
                    context = ssl.create_default_context()
                    with socket.create_connection((hostname, port), timeout=10) as sock:
                        with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                            cert = ssock.getpeercert()
                            
                            # Parse certificate details
                            import datetime
                            not_after = datetime.datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                            not_before = datetime.datetime.strptime(cert['notBefore'], '%b %d %H:%M:%S %Y %Z')
                            
                            days_until_expiry = (not_after - datetime.datetime.utcnow()).days
                            
                            broker_ssl_results[endpoint_type] = {
                                'hostname': hostname,
                                'valid': True,
                                'issuer': cert.get('issuer'),
                                'subject': cert.get('subject'),
                                'serial_number': cert.get('serialNumber'),
                                'not_before': not_before.isoformat(),
                                'not_after': not_after.isoformat(),
                                'days_until_expiry': days_until_expiry,
                                'expired': days_until_expiry < 0
                            }
                
                except Exception as e:
                    broker_ssl_results[endpoint_type] = {
                        'hostname': parsed.hostname,
                        'valid': False,
                        'error': str(e)
                    }
            
            ssl_results[broker] = broker_ssl_results
        
        # Check overall SSL health
        all_valid = all(
            endpoint_result.get('valid', False) and not endpoint_result.get('expired', True)
            for broker_results in ssl_results.values()
            for endpoint_result in broker_results.values()
        )
        
        return {
            'success': all_valid,
            'message': f'SSL certificate validation: {"PASSED" if all_valid else "FAILED"}',
            'ssl_results': ssl_results
        }

    async def test_dns_resolution(self) -> Dict:
        """Test DNS resolution for brokerage endpoints"""
        dns_results = {}
        
        for broker, config in self.brokerage_endpoints.items():
            broker_dns_results = {}
            
            for endpoint_type, url in config.items():
                if endpoint_type == 'ports':
                    continue
                    
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    hostname = parsed.hostname
                    
                    # Resolve hostname
                    import socket
                    start_time = time.time()
                    ip_addresses = socket.gethostbyname_ex(hostname)
                    resolution_time = time.time() - start_time
                    
                    broker_dns_results[endpoint_type] = {
                        'hostname': hostname,
                        'resolved': True,
                        'ip_addresses': ip_addresses[2],
                        'canonical_name': ip_addresses[0],
                        'resolution_time_ms': round(resolution_time * 1000, 2)
                    }
                    
                except Exception as e:
                    broker_dns_results[endpoint_type] = {
                        'hostname': parsed.hostname,
                        'resolved': False,
                        'error': str(e)
                    }
            
            dns_results[broker] = broker_dns_results
        
        # Check overall DNS health
        all_resolved = all(
            endpoint_result.get('resolved', False)
            for broker_results in dns_results.values()
            for endpoint_result in broker_results.values()
        )
        
        return {
            'success': all_resolved,
            'message': f'DNS resolution: {"PASSED" if all_resolved else "FAILED"}',
            'dns_results': dns_results
        }

    async def test_api_reachability(self) -> Dict:
        """Test API endpoint reachability with HTTP requests"""
        api_results = {}
        
        for broker, config in self.brokerage_endpoints.items():
            broker_api_results = {}
            
            for endpoint_type, url in config.items():
                if endpoint_type == 'ports':
                    continue
                    
                try:
                    # Make HTTP request to test reachability
                    start_time = time.time()
                    response = requests.get(
                        url, 
                        timeout=10,
                        headers={'User-Agent': 'MIP-Connectivity-Test/1.0'},
                        verify=True
                    )
                    response_time = time.time() - start_time
                    
                    broker_api_results[endpoint_type] = {
                        'url': url,
                        'reachable': True,
                        'status_code': response.status_code,
                        'response_time_ms': round(response_time * 1000, 2),
                        'headers': dict(response.headers),
                        'ssl_verified': True
                    }
                    
                except requests.exceptions.SSLError as e:
                    broker_api_results[endpoint_type] = {
                        'url': url,
                        'reachable': False,
                        'error': 'SSL Error',
                        'error_details': str(e)
                    }
                except requests.exceptions.RequestException as e:
                    broker_api_results[endpoint_type] = {
                        'url': url,
                        'reachable': False,
                        'error': 'Request Error',
                        'error_details': str(e)
                    }
                except Exception as e:
                    broker_api_results[endpoint_type] = {
                        'url': url,
                        'reachable': False,
                        'error': 'General Error',
                        'error_details': str(e)
                    }
            
            api_results[broker] = broker_api_results
        
        # Calculate success rate
        total_apis = sum(len(broker_results) for broker_results in api_results.values())
        reachable_apis = sum(
            1 for broker_results in api_results.values()
            for endpoint_result in broker_results.values()
            if endpoint_result.get('reachable', False)
        )
        
        return {
            'success': reachable_apis > 0,
            'message': f'API reachability: {reachable_apis}/{total_apis} endpoints reachable',
            'api_results': api_results
        }

    async def test_oauth_endpoints(self) -> Dict:
        """Test OAuth endpoint accessibility for brokerage authentication"""
        oauth_results = {}
        
        oauth_endpoints = {
            'td_ameritrade': 'https://auth.tdameritrade.com/auth',
            'schwab': 'https://api.schwab.com/oauth/authorize'
        }
        
        for broker, oauth_url in oauth_endpoints.items():
            try:
                # Test OAuth endpoint reachability
                response = requests.get(
                    oauth_url,
                    timeout=10,
                    headers={'User-Agent': 'MIP-OAuth-Test/1.0'},
                    allow_redirects=False  # Don't follow redirects for OAuth endpoints
                )
                
                oauth_results[broker] = {
                    'url': oauth_url,
                    'reachable': True,
                    'status_code': response.status_code,
                    'expected_redirect': 300 <= response.status_code < 400,
                    'headers': dict(response.headers)
                }
                
            except Exception as e:
                oauth_results[broker] = {
                    'url': oauth_url,
                    'reachable': False,
                    'error': str(e)
                }
        
        # OAuth endpoints should return redirect status codes
        oauth_working = all(
            result.get('reachable', False) and (
                result.get('expected_redirect', False) or 
                result.get('status_code') == 200
            )
            for result in oauth_results.values()
        )
        
        return {
            'success': oauth_working,
            'message': f'OAuth endpoints: {"WORKING" if oauth_working else "ISSUES_DETECTED"}',
            'oauth_results': oauth_results
        }

    def generate_summary(self):
        """Generate test execution summary"""
        total_tests = len(self.test_results['tests'])
        passed_tests = sum(1 for test in self.test_results['tests'].values() if test['status'] == 'PASSED')
        failed_tests = sum(1 for test in self.test_results['tests'].values() if test['status'] == 'FAILED')
        error_tests = sum(1 for test in self.test_results['tests'].values() if test['status'] == 'ERROR')
        
        self.test_results['summary'] = {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'errors': error_tests,
            'success_rate': round((passed_tests / total_tests) * 100, 2) if total_tests > 0 else 0,
            'overall_status': 'PASSED' if failed_tests == 0 and error_tests == 0 else 'FAILED'
        }

    def save_results(self, filename: str = None):
        """Save test results to JSON file"""
        if not filename:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f'brokerage_connectivity_test_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        logger.info(f"📄 Test results saved to: {filename}")
        return filename


async def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test brokerage connectivity')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--environment', default='dev', help='Environment (dev/staging/prod)')
    parser.add_argument('--output', help='Output file for results')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run tests
    tester = BrokerageConnectivityTester(region=args.region, environment=args.environment)
    results = await tester.run_all_tests()
    
    # Print summary
    summary = results['summary']
    print(f"\n🎯 BROKERAGE CONNECTIVITY TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Environment: {args.environment}")
    print(f"Region: {args.region}")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"✅ Passed: {summary['passed']}")
    print(f"❌ Failed: {summary['failed']}")
    print(f"⚠️  Errors: {summary['errors']}")
    print(f"Success Rate: {summary['success_rate']}%")
    print(f"Overall Status: {summary['overall_status']}")
    
    # Save results
    output_file = tester.save_results(args.output)
    
    # Exit with appropriate code
    exit_code = 0 if summary['overall_status'] == 'PASSED' else 1
    print(f"\n🏁 Test completed with exit code: {exit_code}")
    return exit_code


if __name__ == "__main__":
    import sys
    
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        sys.exit(1)
