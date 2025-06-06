import { create } from 'zustand';
import { WebSocketState, WebSocketMessage } from '@/types';
import { webSocketService } from '@/services/webSocketService';

export const useWebSocketStore = create<WebSocketState>((set, get) => ({
  isConnected: false,
  connectionStatus: 'disconnected',
  lastMessage: null,
  error: null,

  connect: () => {
    set({ connectionStatus: 'connecting', error: null });
    
    try {
      webSocketService.connect();
    } catch (error: any) {
      set({ 
        connectionStatus: 'error', 
        error: error.message || 'Connection failed' 
      });
    }
  },

  disconnect: () => {
    webSocketService.disconnect();
    set({ 
      isConnected: false, 
      connectionStatus: 'disconnected',
      error: null 
    });
  },

  sendMessage: (message: any) => {
    try {
      webSocketService.send(message);
    } catch (error: any) {
      set({ error: error.message || 'Failed to send message' });
    }
  },

  subscribe: (type: string, target: string) => {
    try {
      webSocketService.subscribe(type, target);
    } catch (error: any) {
      set({ error: error.message || 'Failed to subscribe' });
    }
  },

  unsubscribe: (type: string, target: string) => {
    try {
      webSocketService.unsubscribe(type, target);
    } catch (error: any) {
      set({ error: error.message || 'Failed to unsubscribe' });
    }
  },

  updateConnectionStatus: (status: WebSocketState['connectionStatus']) => {
    set({ 
      connectionStatus: status,
      isConnected: status === 'connected',
      error: status === 'error' ? 'Connection error' : null
    });
  },

  updateLastMessage: (message: WebSocketMessage) => {
    set({ lastMessage: message });
  },

  clearError: () => set({ error: null }),
}));

// Initialize WebSocket event listeners
webSocketService.on('connect', () => {
  useWebSocketStore.getState().updateConnectionStatus('connected');
});

webSocketService.on('disconnect', () => {
  useWebSocketStore.getState().updateConnectionStatus('disconnected');
});

webSocketService.on('error', (error: any) => {
  useWebSocketStore.setState({ 
    connectionStatus: 'error',
    isConnected: false,
    error: error.message || 'WebSocket error'
  });
});

webSocketService.on('message', (message: WebSocketMessage) => {
  useWebSocketStore.getState().updateLastMessage(message);
});
