#!/usr/bin/env python3
"""
=============================================================================
TEST DATA DOWNLOAD SCRIPT
Downloads and prepares test data for MIP Platform agents
=============================================================================
"""

import os
import sys
import json
import csv
import requests
import zipfile
import gzip
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import time
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestDataDownloader:
    def __init__(self, base_dir: str = "test-data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Data source URLs
        self.data_sources = {
            'finbert': {
                'financial_phrasebank': 'https://www.researchgate.net/profile/Pekka_Malo/publication/251231364_FinancialPhraseBank-v10/data/0c96051eee9b05e9329dc30a9267ce01/FinancialPhraseBank-v10.zip',
                'sentiment_samples': 'https://raw.githubusercontent.com/ProsusAI/finBERT/master/data/sentiment_data.csv',
                'news_headlines': 'https://github.com/datasets/s-and-p-500-companies/raw/master/data/constituents.csv'
            },
            'llama': {
                'financial_qa': 'https://huggingface.co/datasets/ChanceFocus/flare-finqa/resolve/main/train.jsonl',
                'reasoning_tests': 'https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/logical_deduction/task.json',
                'market_analysis': 'https://raw.githubusercontent.com/microsoft/qlib/main/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml'
            },
            'gpt4': {
                'analysis_prompts': 'https://raw.githubusercontent.com/microsoft/guidance/main/guidance/chat.py',
                'financial_documents': 'https://www.sec.gov/Archives/edgar/daily-index/2023/QTR4/master.20231231.idx'
            },
            'tft': {
                'stock_prices': 'https://query1.finance.yahoo.com/v7/finance/download/SPY?period1=1609459200&period2=1672531200&interval=1d&events=history',
                'economic_indicators': 'https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=GDP&scale=left&cosd=1947-01-01&coed=2023-01-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Quarterly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2023-01-01&revision_date=2023-01-01&nd=1947-01-01',
                'market_volatility': 'https://query1.finance.yahoo.com/v7/finance/download/%5EVIX?period1=1609459200&period2=1672531200&interval=1d&events=history'
            },
            'common': {
                'sp500_companies': 'https://datahub.io/core/s-and-p-500-companies/r/constituents.csv',
                'sector_data': 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents-financials.csv'
            }
        }
    
    def download_file(self, url: str, destination: Path, timeout: int = 30) -> bool:
        """Download a file from URL to destination."""
        try:
            logger.info(f"Downloading {url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Create parent directories if they don't exist
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Download with progress indication
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            if downloaded % (1024 * 1024) == 0:  # Log every MB
                                logger.info(f"Progress: {progress:.1f}%")
            
            logger.info(f"Successfully downloaded {destination.name}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download {url}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error downloading {url}: {e}")
            return False
    
    def extract_archive(self, archive_path: Path, extract_to: Path) -> bool:
        """Extract ZIP or GZ archives."""
        try:
            extract_to.mkdir(parents=True, exist_ok=True)
            
            if archive_path.suffix.lower() == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
                logger.info(f"Extracted ZIP archive to {extract_to}")
                
            elif archive_path.suffix.lower() == '.gz':
                output_file = extract_to / archive_path.stem
                with gzip.open(archive_path, 'rb') as gz_file:
                    with open(output_file, 'wb') as out_file:
                        out_file.write(gz_file.read())
                logger.info(f"Extracted GZ archive to {output_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract {archive_path}: {e}")
            return False
    
    def create_mock_data(self, agent_type: str, data_type: str, output_path: Path) -> bool:
        """Create mock data when downloads fail."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if agent_type == 'finbert':
                if data_type == 'sentiment_samples':
                    mock_data = [
                        {'text': 'Company reports strong quarterly earnings growth', 'sentiment': 'positive', 'confidence': 0.92},
                        {'text': 'Market volatility continues amid economic uncertainty', 'sentiment': 'negative', 'confidence': 0.78},
                        {'text': 'Merger talks between tech companies ongoing', 'sentiment': 'neutral', 'confidence': 0.65},
                        {'text': 'Stock buyback program announced by corporation', 'sentiment': 'positive', 'confidence': 0.84},
                        {'text': 'Regulatory investigation impacts share price', 'sentiment': 'negative', 'confidence': 0.89},
                        {'text': 'Quarterly results meet analyst expectations', 'sentiment': 'neutral', 'confidence': 0.71},
                        {'text': 'New product launch exceeds sales forecasts', 'sentiment': 'positive', 'confidence': 0.95},
                        {'text': 'Supply chain disruptions affect operations', 'sentiment': 'negative', 'confidence': 0.82}
                    ]
                elif data_type == 'news_headlines':
                    mock_data = [
                        {'symbol': 'AAPL', 'headline': 'Apple announces new iPhone with advanced features'},
                        {'symbol': 'MSFT', 'headline': 'Microsoft cloud revenue shows strong growth'},
                        {'symbol': 'GOOGL', 'headline': 'Google faces antitrust investigation'},
                        {'symbol': 'AMZN', 'headline': 'Amazon expands logistics network'},
                        {'symbol': 'TSLA', 'headline': 'Tesla deliveries exceed expectations'}
                    ]
                else:
                    mock_data = {'note': f'Mock data for {data_type}', 'samples': 100}
            
            elif agent_type == 'llama':
                if data_type == 'reasoning_tests':
                    mock_data = {
                        'examples': [
                            {
                                'input': 'If a company has P/E ratio of 15 and industry average is 20, what does this suggest?',
                                'target': 'The company appears undervalued relative to industry peers',
                                'reasoning': 'Lower P/E ratio suggests the stock is trading at a discount'
                            },
                            {
                                'input': 'Revenue increased 20% but operating margin decreased from 15% to 12%. Analyze the situation.',
                                'target': 'Revenue growth is positive but profitability efficiency declined',
                                'reasoning': 'Cost growth outpaced revenue growth, indicating operational challenges'
                            }
                        ]
                    }
                elif data_type == 'financial_qa':
                    mock_data = [
                        {'question': 'What is EBITDA?', 'answer': 'Earnings Before Interest, Taxes, Depreciation, and Amortization'},
                        {'question': 'How do you calculate ROE?', 'answer': 'Net Income divided by Shareholders Equity'},
                        {'question': 'What is a bull market?', 'answer': 'A market characterized by rising prices and investor optimism'}
                    ]
                else:
                    mock_data = {'reasoning_scenarios': [f'Scenario {i}' for i in range(1, 11)]}
            
            elif agent_type == 'tft':
                if data_type == 'stock_prices':
                    # Generate mock price data
                    base_price = 100.0
                    dates = []
                    prices = []
                    
                    start_date = datetime.now() - timedelta(days=252)  # 1 year of trading days
                    current_price = base_price
                    
                    for i in range(252):
                        date = start_date + timedelta(days=i)
                        if date.weekday() < 5:  # Only weekdays
                            # Random walk with slight upward drift
                            change = random.gauss(0.001, 0.02)  # 0.1% drift, 2% volatility
                            current_price *= (1 + change)
                            
                            dates.append(date.strftime('%Y-%m-%d'))
                            prices.append(round(current_price, 2))
                    
                    mock_data = []
                    for date, price in zip(dates, prices):
                        mock_data.append({
                            'Date': date,
                            'Open': round(price * random.uniform(0.99, 1.01), 2),
                            'High': round(price * random.uniform(1.00, 1.03), 2),
                            'Low': round(price * random.uniform(0.97, 1.00), 2),
                            'Close': price,
                            'Volume': random.randint(1000000, 5000000)
                        })
                
                elif data_type == 'market_volatility':
                    # Generate VIX-like volatility data
                    mock_data = []
                    base_vix = 20.0
                    
                    for i in range(252):
                        date = (datetime.now() - timedelta(days=252-i)).strftime('%Y-%m-%d')
                        vix_value = max(10, base_vix + random.gauss(0, 5))  # VIX can't be negative
                        mock_data.append({
                            'Date': date,
                            'VIX': round(vix_value, 2)
                        })
                
                else:
                    mock_data = {'time_series_length': 252, 'features': ['Open', 'High', 'Low', 'Close', 'Volume']}
            
            else:  # Common data
                if data_type == 'sp500_companies':
                    mock_data = [
                        {'Symbol': 'AAPL', 'Name': 'Apple Inc.', 'Sector': 'Information Technology'},
                        {'Symbol': 'MSFT', 'Name': 'Microsoft Corporation', 'Sector': 'Information Technology'},
                        {'Symbol': 'AMZN', 'Name': 'Amazon.com Inc.', 'Sector': 'Consumer Discretionary'},
                        {'Symbol': 'GOOGL', 'Name': 'Alphabet Inc.', 'Sector': 'Communication Services'},
                        {'Symbol': 'TSLA', 'Name': 'Tesla Inc.', 'Sector': 'Consumer Discretionary'}
                    ]
                else:
                    mock_data = {'note': f'Mock data for {data_type}'}
            
            # Write mock data
            if output_path.suffix.lower() == '.csv':
                if isinstance(mock_data, list) and len(mock_data) > 0:
                    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                        fieldnames = mock_data[0].keys()
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(mock_data)
                else:
                    # Fallback for non-list data
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write("# Mock CSV data\n")
                        f.write(json.dumps(mock_data, indent=2))
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(mock_data, f, indent=2)
            
            logger.info(f"Created mock data: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create mock data for {agent_type}/{data_type}: {e}")
            return False
    
    def download_agent_data(self, agent_type: str) -> bool:
        """Download all data for a specific agent."""
        if agent_type not in self.data_sources:
            logger.error(f"Unknown agent type: {agent_type}")
            return False
        
        agent_dir = self.base_dir / agent_type
        agent_dir.mkdir(exist_ok=True)
        
        sources = self.data_sources[agent_type]
        success_count = 0
        
        for data_name, url in sources.items():
            # Determine file extension from URL
            if url.endswith('.csv'):
                file_ext = '.csv'
            elif url.endswith('.json') or url.endswith('.jsonl'):
                file_ext = '.json'
            elif url.endswith('.zip'):
                file_ext = '.zip'
            elif url.endswith('.gz'):
                file_ext = '.gz'
            else:
                file_ext = '.txt'
            
            file_path = agent_dir / f"{data_name}{file_ext}"
            
            # Try to download
            if self.download_file(url, file_path):
                # Extract if it's an archive
                if file_ext in ['.zip', '.gz']:
                    extract_dir = agent_dir / data_name
                    self.extract_archive(file_path, extract_dir)
                
                success_count += 1
            else:
                # Create mock data if download fails
                logger.warning(f"Download failed for {data_name}, creating mock data")
                mock_file_path = agent_dir / f"{data_name}_mock.json"
                if self.create_mock_data(agent_type, data_name, mock_file_path):
                    success_count += 1
        
        success_rate = success_count / len(sources)
        logger.info(f"Agent {agent_type}: {success_count}/{len(sources)} datasets acquired ({success_rate:.1%})")
        
        return success_rate > 0.5  # Consider successful if more than half succeeded
    
    def download_all_data(self) -> Dict[str, bool]:
        """Download data for all agents."""
        results = {}
        
        # Download common data first
        logger.info("Downloading common datasets...")
        results['common'] = self.download_agent_data('common')
        
        # Download agent-specific data
        for agent_type in ['finbert', 'llama', 'gpt4', 'tft']:
            logger.info(f"Downloading data for {agent_type} agent...")
            results[agent_type] = self.download_agent_data(agent_type)
            
            # Add a small delay between downloads to be respectful
            time.sleep(1)
        
        return results
    
    def validate_data(self, agent_type: str) -> Dict[str, Any]:
        """Validate downloaded data for an agent."""
        agent_dir = self.base_dir / agent_type
        if not agent_dir.exists():
            return {'status': 'error', 'message': f'No data directory for {agent_type}'}
        
        files = list(agent_dir.glob('*'))
        validation_results = {
            'status': 'success',
            'agent': agent_type,
            'files_found': len(files),
            'file_details': [],
            'total_size_mb': 0
        }
        
        for file_path in files:
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                validation_results['total_size_mb'] += size_mb
                
                file_info = {
                    'name': file_path.name,
                    'size_mb': round(size_mb, 2),
                    'type': file_path.suffix
                }
                
                # Basic content validation
                try:
                    if file_path.suffix.lower() == '.json':
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            file_info['records'] = len(data) if isinstance(data, list) else 1
                    elif file_path.suffix.lower() == '.csv':
                        with open(file_path, 'r') as f:
                            reader = csv.reader(f)
                            file_info['records'] = sum(1 for row in reader) - 1  # Exclude header
                    
                    file_info['status'] = 'valid'
                except Exception as e:
                    file_info['status'] = 'invalid'
                    file_info['error'] = str(e)
                
                validation_results['file_details'].append(file_info)
        
        validation_results['total_size_mb'] = round(validation_results['total_size_mb'], 2)
        return validation_results

def main():
    parser = argparse.ArgumentParser(description='Download test data for MIP Platform agents')
    parser.add_argument('agent_type', nargs='?', choices=['finbert', 'llama', 'gpt4', 'tft', 'common', 'all'],
                       default='all', help='Agent type to download data for (default: all)')
    parser.add_argument('--output-dir', '-o', default='test-data',
                       help='Output directory for downloaded data (default: test-data)')
    parser.add_argument('--validate', '-v', action='store_true',
                       help='Validate downloaded data')
    parser.add_argument('--force', '-f', action='store_true',
                       help='Force re-download even if files exist')
    parser.add_argument('--mock-only', action='store_true',
                       help='Create only mock data (skip downloads)')
    parser.add_argument('--timeout', '-t', type=int, default=60,
                       help='Download timeout in seconds (default: 60)')
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = TestDataDownloader(args.output_dir)
    
    try:
        if args.agent_type == 'all':
            if args.mock_only:
                logger.info("Creating mock data for all agents...")
                results = {}
                for agent in ['finbert', 'llama', 'gpt4', 'tft', 'common']:
                    agent_dir = downloader.base_dir / agent
                    agent_dir.mkdir(exist_ok=True)
                    
                    # Create mock data for each agent's expected datasets
                    sources = downloader.data_sources.get(agent, {})
                    for data_name in sources.keys():
                        mock_file = agent_dir / f"{data_name}_mock.json"
                        results[f"{agent}_{data_name}"] = downloader.create_mock_data(agent, data_name, mock_file)
            else:
                logger.info("Downloading data for all agents...")
                results = downloader.download_all_data()
        else:
            if args.mock_only:
                logger.info(f"Creating mock data for {args.agent_type}...")
                agent_dir = downloader.base_dir / args.agent_type
                agent_dir.mkdir(exist_ok=True)
                
                sources = downloader.data_sources.get(args.agent_type, {})
                for data_name in sources.keys():
                    mock_file = agent_dir / f"{data_name}_mock.json"
                    downloader.create_mock_data(args.agent_type, data_name, mock_file)
                results = {args.agent_type: True}
            else:
                logger.info(f"Downloading data for {args.agent_type}...")
                results = {args.agent_type: downloader.download_agent_data(args.agent_type)}
        
        # Print results summary
        logger.info("\n" + "="*50)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("="*50)
        
        for agent, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            logger.info(f"{agent:15} {status}")
        
        # Validation if requested
        if args.validate:
            logger.info("\n" + "="*50)
            logger.info("VALIDATION RESULTS")
            logger.info("="*50)
            
            if args.agent_type == 'all':
                for agent in ['finbert', 'llama', 'gpt4', 'tft', 'common']:
                    validation = downloader.validate_data(agent)
                    logger.info(f"\n{agent.upper()} Agent:")
                    logger.info(f"  Files: {validation['files_found']}")
                    logger.info(f"  Total Size: {validation['total_size_mb']} MB")
                    for file_detail in validation['file_details']:
                        logger.info(f"    {file_detail['name']}: {file_detail.get('records', 'N/A')} records, {file_detail['size_mb']} MB")
            else:
                validation = downloader.validate_data(args.agent_type)
                print(json.dumps(validation, indent=2))
        
        # Return appropriate exit code
        overall_success = all(results.values()) if results else False
        if not overall_success:
            logger.error("Some downloads failed")
            sys.exit(1)
        
        logger.info("All downloads completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
