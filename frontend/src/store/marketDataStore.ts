import { create } from 'zustand';
import { MarketDataState, MarketQuote } from '@/types';
import { apiService } from '@/services/apiService';
import { webSocketService } from '@/services/webSocketService';
import toast from 'react-hot-toast';

export const useMarketDataStore = create<MarketDataState>((set, get) => ({
  quotes: {},
  subscriptions: new Set(),
  isLoading: false,
  error: null,

  subscribe: (symbols: string[]) => {
    const { subscriptions } = get();
    const newSymbols = symbols.filter(symbol => !subscriptions.has(symbol));
    
    if (newSymbols.length === 0) return;

    set({ isLoading: true, error: null });
    
    try {
      // Subscribe via WebSocket
      newSymbols.forEach(symbol => {
        webSocketService.subscribe('market_data', symbol);
      });
      
      // Update subscriptions
      const newSubscriptions = new Set([...subscriptions, ...newSymbols]);
      set({ subscriptions: newSubscriptions, isLoading: false });
      
      // Fetch initial data
      get().fetchQuotes(newSymbols);
      
    } catch (error: any) {
      const errorMessage = 'Failed to subscribe to market data';
      set({ 
        isLoading: false, 
        error: errorMessage 
      });
      toast.error(errorMessage);
    }
  },

  unsubscribe: (symbols: string[]) => {
    const { subscriptions } = get();
    
    try {
      // Unsubscribe via WebSocket
      symbols.forEach(symbol => {
        webSocketService.unsubscribe('market_data', symbol);
      });
      
      // Update subscriptions
      const newSubscriptions = new Set(subscriptions);
      symbols.forEach(symbol => newSubscriptions.delete(symbol));
      
      // Remove quotes for unsubscribed symbols
      const newQuotes = { ...get().quotes };
      symbols.forEach(symbol => delete newQuotes[symbol]);
      
      set({ 
        subscriptions: newSubscriptions,
        quotes: newQuotes
      });
      
    } catch (error: any) {
      console.error('Failed to unsubscribe from market data:', error);
    }
  },

  getQuote: (symbol: string): MarketQuote | null => {
    return get().quotes[symbol] || null;
  },

  fetchQuotes: async (symbols: string[]) => {
    try {
      const symbolsParam = symbols.join(',');
      const response = await apiService.get<{ prices: { [symbol: string]: number }; timestamp: string }>(
        `/market/prices?symbols=${symbolsParam}`
      );
      
      // Convert price data to quotes format
      const newQuotes: { [symbol: string]: MarketQuote } = {};
      Object.entries(response.data.prices).forEach(([symbol, price]) => {
        newQuotes[symbol] = {
          symbol,
          price,
          bid: price - 0.01,
          ask: price + 0.01,
          volume: 0,
          change: 0,
          change_percent: 0,
          high: price,
          low: price,
          open_price: price,
          timestamp: response.data.timestamp
        };
      });
      
      set(state => ({
        quotes: { ...state.quotes, ...newQuotes }
      }));
      
    } catch (error: any) {
      console.error('Failed to fetch quotes:', error);
    }
  },

  updateQuote: (quote: MarketQuote) => {
    set(state => ({
      quotes: {
        ...state.quotes,
        [quote.symbol]: quote
      }
    }));
  },

  clearError: () => set({ error: null }),
}));

// Initialize market data WebSocket listeners
webSocketService.on('price_update', (data: any) => {
  const quote: MarketQuote = {
    symbol: data.symbol,
    price: data.price,
    bid: data.price - 0.01,
    ask: data.price + 0.01,
    volume: data.volume || 0,
    change: data.change || 0,
    change_percent: data.change_percent || 0,
    high: data.high || data.price,
    low: data.low || data.price,
    open_price: data.open_price || data.price,
    timestamp: data.timestamp
  };
  
  useMarketDataStore.getState().updateQuote(quote);
});
