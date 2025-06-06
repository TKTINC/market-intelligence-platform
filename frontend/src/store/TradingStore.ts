import { create } from 'zustand';
import { TradingState, TradeRequest, TradeResponse, TradeHistory } from '@/types';
import { apiService } from '@/services/apiService';
import toast from 'react-hot-toast';

export const useTradingStore = create<TradingState>((set, get) => ({
  trades: [],
  pendingTrades: [],
  isLoading: false,
  error: null,

  executeTrade: async (trade: TradeRequest): Promise<TradeResponse> => {
    set({ isLoading: true, error: null });
    
    try {
      const response = await apiService.post<TradeResponse>('/trading/execute', trade);
      const executedTrade = response.data;
      
      set(state => ({
        trades: [executedTrade, ...state.trades],
        pendingTrades: state.pendingTrades.filter(pt => 
          !(pt.portfolio_id === trade.portfolio_id && pt.symbol === trade.symbol)
        ),
        isLoading: false
      }));
      
      const actionText = trade.action === 'buy' ? 'Bought' : trade.action === 'sell' ? 'Sold' : 'Closed';
      toast.success(
        `${actionText} ${trade.quantity} shares of ${trade.symbol} at ${executedTrade.executed_price.toFixed(2)}`
      );
      
      return executedTrade;
      
    } catch (error: any) {
      const errorMessage = error.response?.data?.message || 'Trade execution failed';
      set({ 
        isLoading: false, 
        error: errorMessage 
      });
      toast.error(errorMessage);
      throw error;
    }
  },

  fetchTradeHistory: async (portfolioId: string) => {
    set({ isLoading: true, error: null });
    
    try {
      const response = await apiService.get<TradeHistory>(`/trading/history/${portfolioId}`);
      
      set({ 
        trades: response.data.trades,
        isLoading: false 
      });
      
    } catch (error: any) {
      const errorMessage = error.response?.data?.message || 'Failed to fetch trade history';
      set({ 
        isLoading: false, 
        error: errorMessage 
      });
      toast.error(errorMessage);
    }
  },

  cancelTrade: async (tradeId: string) => {
    set({ isLoading: true, error: null });
    
    try {
      await apiService.delete(`/trading/cancel/${tradeId}`);
      
      set(state => ({
        pendingTrades: state.pendingTrades.filter(pt => pt.trade_id !== tradeId),
        isLoading: false
      }));
      
      toast.success('Trade cancelled successfully');
      
    } catch (error: any) {
      const errorMessage = error.response?.data?.message || 'Failed to cancel trade';
      set({ 
        isLoading: false, 
        error: errorMessage 
      });
      toast.error(errorMessage);
      throw error;
    }
  },

  clearError: () => set({ error: null }),
}));
