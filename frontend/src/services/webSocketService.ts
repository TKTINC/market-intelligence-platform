import { WebSocketMessage } from '@/types';

type EventHandler = (data: any) => void;

class WebSocketService {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectInterval = 1000;
  private eventHandlers: Map<string, EventHandler[]> = new Map();
  private subscriptions: Set<string> = new Set();
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private isConnecting = false;

  constructor() {
    this.url = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws';
  }

  connect(userId?: string): void {
    if (this.isConnecting || (this.ws && this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      return;
    }

    this.isConnecting = true;
    
    try {
      // Connect to user-specific WebSocket endpoint if userId provided
      const wsUrl = userId ? `${this.url}/user/${userId}` : `${this.url}/general`;
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = this.handleOpen.bind(this);
      this.ws.onmessage = this.handleMessage.bind(this);
      this.ws.onclose = this.handleClose.bind(this);
      this.ws.onerror = this.handleError.bind(this);

    } catch (error) {
      this.isConnecting = false;
      this.emit('error', error);
    }
  }

  disconnect(): void {
    this.clearHeartbeat();
    this.reconnectAttempts = 0;
    
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    
    this.isConnecting = false;
    this.emit('disconnect');
  }

  send(message: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      throw new Error('WebSocket is not connected');
    }
  }

  subscribe(type: string, target: string): void {
    const subscriptionKey = `${type}:${target}`;
    
    if (this.subscriptions.has(subscriptionKey)) {
      return;
    }

    this.subscriptions.add(subscriptionKey);

    this.send({
      type: 'subscribe',
      subscription_type: type,
      target: target,
    });
  }

  unsubscribe(type: string, target: string): void {
    const subscriptionKey = `${type}:${target}`;
    
    if (!this.subscriptions.has(subscriptionKey)) {
      return;
    }

    this.subscriptions.delete(subscriptionKey);

    this.send({
      type: 'unsubscribe',
      subscription_type: type,
      target: target,
    });
  }

  on(event: string, handler: EventHandler): void {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, []);
    }
    this.eventHandlers.get(event)!.push(handler);
  }

  off(event: string, handler: EventHandler): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    }
  }

  private emit(event: string, data?: any): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(data);
        } catch (error) {
          console.error(`Error in WebSocket event handler for ${event}:`, error);
        }
      });
    }
  }

  private handleOpen(event: Event): void {
    console.log('WebSocket connected');
    this.isConnecting = false;
    this.reconnectAttempts = 0;
    
    // Start heartbeat
    this.startHeartbeat();
    
    // Resubscribe to all previous subscriptions
    this.subscriptions.forEach(subscription => {
      const [type, target] = subscription.split(':');
      this.send({
        type: 'subscribe',
        subscription_type: type,
        target: target,
      });
    });

    this.emit('connect');
  }

  private handleMessage(event: MessageEvent): void {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);
      
      // Handle different message types
      switch (message.type) {
        case 'pong':
          // Handle pong response
          break;
          
        case 'error':
          console.error('WebSocket error:', message.data);
          this.emit('error', message.data);
          break;
          
        case 'subscription_confirmed':
          console.log('Subscription confirmed:', message.data);
          break;
          
        case 'price_update':
        case 'portfolio_update':
        case 'trade_executed':
        case 'analysis_result':
        case 'risk_alert':
          this.emit(message.type, message.data);
          break;
          
        default:
          console.log('Unknown message type:', message.type);
      }

      this.emit('message', message);
      
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  }

  private handleClose(event: CloseEvent): void {
    console.log('WebSocket disconnected:', event.code, event.reason);
    this.isConnecting = false;
    this.clearHeartbeat();
    this.ws = null;

    this.emit('disconnect', { code: event.code, reason: event.reason });

    // Attempt to reconnect if not intentionally closed
    if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
      this.scheduleReconnect();
    }
  }

  private handleError(event: Event): void {
    console.error('WebSocket error:', event);
    this.isConnecting = false;
    this.emit('error', event);
  }

  private scheduleReconnect(): void {
    this.reconnectAttempts++;
    const delay = this.reconnectInterval * Math.pow(2, this.reconnectAttempts - 1);
    
    console.log(`Scheduling WebSocket reconnect attempt ${this.reconnectAttempts} in ${delay}ms`);
    
    setTimeout(() => {
      if (this.reconnectAttempts <= this.maxReconnectAttempts) {
        this.connect();
      }
    }, delay);
  }

  private startHeartbeat(): void {
    this.clearHeartbeat();
    
    this.heartbeatInterval = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.send({ type: 'ping', timestamp: new Date().toISOString() });
      }
    }, 30000); // Send ping every 30 seconds
  }

  private clearHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  // Utility methods
  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  getReadyState(): number | null {
    return this.ws ? this.ws.readyState : null;
  }

  getConnectionStatus(): string {
    if (!this.ws) return 'disconnected';
    
    switch (this.ws.readyState) {
      case WebSocket.CONNECTING:
        return 'connecting';
      case WebSocket.OPEN:
        return 'connected';
      case WebSocket.CLOSING:
        return 'closing';
      case WebSocket.CLOSED:
        return 'disconnected';
      default:
        return 'unknown';
    }
  }
}

export const webSocketService = new WebSocketService();

// Auto-connect when auth state changes
export const initializeWebSocket = (userId?: string) => {
  if (userId) {
    webSocketService.connect(userId);
  }
};
