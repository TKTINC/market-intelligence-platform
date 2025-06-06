import axios, { AxiosInstance, AxiosResponse, AxiosError } from 'axios';
import { APIError } from '@/types';
import toast from 'react-hot-toast';

class APIService {
  private api: AxiosInstance;
  private authToken: string | null = null;

  constructor() {
    this.api = axios.create({
      baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // Request interceptor
    this.api.interceptors.request.use(
      (config) => {
        if (this.authToken) {
          config.headers.Authorization = `Bearer ${this.authToken}`;
        }
        
        // Add request ID for tracking
        config.headers['X-Request-ID'] = this.generateRequestId();
        
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.api.interceptors.response.use(
      (response: AxiosResponse) => {
        return response;
      },
      async (error: AxiosError) => {
        const originalRequest = error.config as any;

        // Handle 401 errors (unauthorized)
        if (error.response?.status === 401 && !originalRequest._retry) {
          originalRequest._retry = true;

          try {
            // Try to refresh token
            const { useAuthStore } = await import('@/store/authStore');
            await useAuthStore.getState().refreshToken();
            
            // Retry original request
            if (this.authToken) {
              originalRequest.headers.Authorization = `Bearer ${this.authToken}`;
            }
            
            return this.api(originalRequest);
          } catch (refreshError) {
            // Refresh failed, redirect to login
            const { useAuthStore } = await import('@/store/authStore');
            useAuthStore.getState().logout();
            window.location.href = '/login';
            return Promise.reject(refreshError);
          }
        }

        // Handle rate limiting (429)
        if (error.response?.status === 429) {
          const retryAfter = error.response.headers['retry-after'];
          const message = `Rate limit exceeded. ${retryAfter ? `Try again in ${retryAfter} seconds.` : 'Please try again later.'}`;
          toast.error(message);
        }

        // Handle network errors
        if (!error.response) {
          toast.error('Network error. Please check your connection.');
        }

        return Promise.reject(this.formatError(error));
      }
    );
  }

  private formatError(error: AxiosError): APIError {
    const response = error.response;
    
    return {
      error: response?.data?.error || 'Request failed',
      message: response?.data?.message || error.message || 'An unexpected error occurred',
      status_code: response?.status || 0,
      timestamp: new Date().toISOString(),
      request_id: response?.headers['x-request-id'],
    };
  }

  private generateRequestId(): string {
    return Math.random().toString(36).substring(2) + Date.now().toString(36);
  }

  setAuthToken(token: string) {
    this.authToken = token;
  }

  clearAuthToken() {
    this.authToken = null;
  }

  // HTTP Methods
  async get<T>(url: string, params?: any): Promise<AxiosResponse<T>> {
    return this.api.get<T>(url, { params });
  }

  async post<T>(url: string, data?: any): Promise<AxiosResponse<T>> {
    return this.api.post<T>(url, data);
  }

  async put<T>(url: string, data?: any): Promise<AxiosResponse<T>> {
    return this.api.put<T>(url, data);
  }

  async patch<T>(url: string, data?: any): Promise<AxiosResponse<T>> {
    return this.api.patch<T>(url, data);
  }

  async delete<T>(url: string): Promise<AxiosResponse<T>> {
    return this.api.delete<T>(url);
  }

  // Specialized methods for file uploads
  async uploadFile<T>(url: string, file: File, onProgress?: (progress: number) => void): Promise<AxiosResponse<T>> {
    const formData = new FormData();
    formData.append('file', file);

    return this.api.post<T>(url, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(progress);
        }
      },
    });
  }

  // Health check method
  async healthCheck(): Promise<boolean> {
    try {
      const response = await this.get('/health');
      return response.status === 200;
    } catch {
      return false;
    }
  }
}

export const apiService = new APIService();
