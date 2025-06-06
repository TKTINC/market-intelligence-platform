import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { AuthState, LoginRequest, AuthTokens, User } from '@/types';
import { apiService } from '@/services/apiService';
import toast from 'react-hot-toast';

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      tokens: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,

      login: async (credentials: LoginRequest) => {
        set({ isLoading: true, error: null });
        
        try {
          const response = await apiService.post<AuthTokens>('/auth/login', credentials);
          const tokens = response.data;
          
          // Store tokens
          set({ 
            tokens,
            user: tokens.user,
            isAuthenticated: true,
            isLoading: false,
            error: null 
          });

          // Set API token for future requests
          apiService.setAuthToken(tokens.access_token);
          
          toast.success(`Welcome back, ${tokens.user.username}!`);
          
        } catch (error: any) {
          const errorMessage = error.response?.data?.message || 'Login failed';
          set({ 
            isLoading: false, 
            error: errorMessage,
            isAuthenticated: false,
            user: null,
            tokens: null
          });
          toast.error(errorMessage);
          throw error;
        }
      },

      logout: () => {
        const { tokens } = get();
        
        if (tokens) {
          // Call logout endpoint
          apiService.post('/auth/logout', { token: tokens.access_token })
            .catch(error => console.error('Logout API call failed:', error));
        }

        // Clear local state
        set({ 
          user: null, 
          tokens: null, 
          isAuthenticated: false, 
          error: null 
        });

        // Clear API token
        apiService.clearAuthToken();
        
        toast.success('Logged out successfully');
      },

      refreshToken: async () => {
        const { tokens } = get();
        
        if (!tokens?.refresh_token) {
          throw new Error('No refresh token available');
        }

        try {
          const response = await apiService.post<Pick<AuthTokens, 'access_token' | 'expires_in'>>('/auth/refresh', {
            refresh_token: tokens.refresh_token
          });

          const newTokens = {
            ...tokens,
            access_token: response.data.access_token,
            expires_in: response.data.expires_in
          };

          set({ tokens: newTokens });
          apiService.setAuthToken(newTokens.access_token);
          
        } catch (error) {
          console.error('Token refresh failed:', error);
          // If refresh fails, logout user
          get().logout();
          throw error;
        }
      },

      clearError: () => set({ error: null }),
    }),
    {
      name: 'mip-auth-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({ 
        tokens: state.tokens, 
        user: state.user, 
        isAuthenticated: state.isAuthenticated 
      }),
    }
  )
);

// Initialize auth on app load
export const initializeAuth = () => {
  const { tokens, isAuthenticated } = useAuthStore.getState();
  
  if (isAuthenticated && tokens?.access_token) {
    apiService.setAuthToken(tokens.access_token);
  }
};
