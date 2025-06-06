import { create } from 'zustand';
import { PortfolioState, Portfolio, Position } from '@/types';
import { apiService } from '@/services/apiService';
import toast from 'react-hot-toast';

export const usePortfolioStore = create<PortfolioState>((set, get) => ({
  portfolios: [],
  selectedPortfolio: null,
  positions: [],
  isLoading: false,
  error: null,

  fetchPortfolios: async () => {
    set({ isLoading: true, error: null });
    
    try {
      const response = await apiService.get<{ portfolios: Portfolio[] }>('/portfolios/user/me');
      
      set({ 
        portfolios: response.data.portfolios,
        isLoading: false 
      });
      
    } catch (error: any) {
      const errorMessage = error.response?.data?.message || 'Failed to fetch portfolios';
      set({ 
        isLoading: false, 
        error: errorMessage 
      });
      toast.error(errorMessage);
    }
  },

  selectPortfolio: async (portfolioId: string) => {
    set({ isLoading: true, error: null });
    
    try {
      const response = await apiService.get<Portfolio>(`/portfolios/${portfolioId}`);
      
      set({ 
        selectedPortfolio: response.data,
        positions: response.data.positions || [],
        isLoading: false 
      });
      
    } catch (error: any) {
      const errorMessage = error.response?.data?.message || 'Failed to fetch portfolio';
      set({ 
        isLoading: false, 
        error: errorMessage 
      });
      toast.error(errorMessage);
    }
  },

  createPortfolio: async (data: {
    name: string;
    initial_balance?: number;
    risk_tolerance?: 'low' | 'medium' | 'high';
    max_position_size?: number;
  }) => {
    set({ isLoading: true, error: null });
    
    try {
      const response = await apiService.post<Portfolio>('/portfolios/create', data);
      
      const newPortfolio = response.data;
      
      set(state => ({ 
        portfolios: [...state.portfolios, newPortfolio],
        isLoading: false 
      }));
      
      toast.success(`Portfolio "${newPortfolio.name}" created successfully`);
      
    } catch (error: any) {
      const errorMessage = error.response?.data?.message || 'Failed to create portfolio';
      set({ 
        isLoading: false, 
        error: errorMessage 
      });
      toast.error(errorMessage);
      throw error;
    }
  },

  updatePortfolio: async (portfolioId: string, data: Partial<Portfolio>) => {
    set({ isLoading: true, error: null });
    
    try {
      const response = await apiService.put<Portfolio>(`/portfolios/${portfolioId}`, data);
      
      const updatedPortfolio = response.data;
      
      set(state => ({
        portfolios: state.portfolios.map(p => 
          p.portfolio_id === portfolioId ? updatedPortfolio : p
        ),
        selectedPortfolio: state.selectedPortfolio?.portfolio_id === portfolioId 
          ? updatedPortfolio 
          : state.selectedPortfolio,
        isLoading: false
      }));
      
      toast.success('Portfolio updated successfully');
      
    } catch (error: any) {
      const errorMessage = error.response?.data?.message || 'Failed to update portfolio';
      set({ 
        isLoading: false, 
        error: errorMessage 
      });
      toast.error(errorMessage);
      throw error;
    }
  },

  deletePortfolio: async (portfolioId: string) => {
    set({ isLoading: true, error: null });
    
    try {
      await apiService.delete(`/portfolios/${portfolioId}`);
      
      set(state => ({
        portfolios: state.portfolios.filter(p => p.portfolio_id !== portfolioId),
        selectedPortfolio: state.selectedPortfolio?.portfolio_id === portfolioId 
          ? null 
          : state.selectedPortfolio,
        positions: state.selectedPortfolio?.portfolio_id === portfolioId 
          ? [] 
          : state.positions,
        isLoading: false
      }));
      
      toast.success('Portfolio deleted successfully');
      
    } catch (error: any) {
      const errorMessage = error.response?.data?.message || 'Failed to delete portfolio';
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
