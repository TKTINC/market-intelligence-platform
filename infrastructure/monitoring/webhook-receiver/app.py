from flask import Flask, request, jsonify
import requests
import json
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

app = Flask(__name__)

# Configuration
SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL')
EMAIL_SMTP_SERVER = os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com')
EMAIL_USERNAME = os.getenv('EMAIL_USERNAME')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')

@app.route('/webhook', methods=['POST'])
def webhook():
    """Receive webhook from AlertManager and process alerts"""
    try:
        data = request.get_json()
        
        for alert in data.get('alerts', []):
            alert_name = alert.get('labels', {}).get('alertname', 'Unknown')
            severity = alert.get('labels', {}).get('severity', 'info')
            
            message = format_alert_message(alert)
            
            if severity in ['critical', 'warning']:
                send_slack_notification(message, severity)
                
            if severity == 'critical':
                send_email_notification(alert_name, message)
                
        return jsonify({'status': 'success'}), 200
        
    except Exception as e:
        print(f"Error processing webhook: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

def format_alert_message(alert):
    """Format alert message for notifications"""
    labels = alert.get('labels', {})
    annotations = alert.get('annotations', {})
    
    message = f"""
🚨 *Alert*: {labels.get('alertname', 'Unknown')}
🔥 *Severity*: {labels.get('severity', 'unknown')}
🏷️ *Service*: {labels.get('service', 'unknown')}
📝 *Summary*: {annotations.get('summary', 'No summary available')}
⏰ *Time*: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
    """.strip()
    
    return message

def send_slack_notification(message, severity):
    """Send notification to Slack"""
    if not SLACK_WEBHOOK_URL:
        return
        
    color_map = {
        'critical': 'danger',
        'warning': 'warning',
        'info': 'good'
    }
    
    payload = {
        'text': message,
        'color': color_map.get(severity, 'good'),
        'username': 'Market Intelligence Monitor'
    }
    
    try:
        requests.post(SLACK_WEBHOOK_URL, json=payload)
    except Exception as e:
        print(f"Failed to send Slack notification: {e}")

def send_email_notification(alert_name, message):
    """Send email notification for critical alerts"""
    if not all([EMAIL_USERNAME, EMAIL_PASSWORD]):
        return
        
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USERNAME
        msg['To'] = 'devops@marketintelligence.com'
        msg['Subject'] = f'🚨 Critical Alert: {alert_name}'
        
        msg.attach(MIMEText(message, 'plain'))
        
        server = smtplib.SMTP(EMAIL_SMTP_SERVER, 587)
        server.starttls()
        server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        
    except Exception as e:
        print(f"Failed to send email: {e}")

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
