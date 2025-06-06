import React, { useEffect } from 'react';
import { Outlet } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import Header from './Header';
import Sidebar from './Sidebar';
import { useAuthStore } from '@/store/authStore';
import { useWebSocketStore } from '@/store/webSocketStore';
import { initializeWebSocket } from '@/services/webSocketService';

const Layout: React.FC = () => {
  const { user, isAuthenticated } = useAuthStore();
  const { connect, isConnected } = useWebSocketStore();

  useEffect(() => {
    if (isAuthenticated && user && !isConnected) {
      // Initialize WebSocket connection with user ID
      initializeWebSocket(user.user_id);
    }
  }, [isAuthenticated, user, isConnected]);

  return (
    <div className="h-screen bg-gray-50 flex overflow-hidden">
      {/* Sidebar */}
      <Sidebar />
      
      {/* Main Content */}
      <div className="flex-1 overflow-hidden flex flex-col">
        <Header />
        
        {/* Page Content */}
        <main className="flex-1 relative overflow-y-auto focus:outline-none">
          <div className="py-6">
            <Outlet />
          </div>
        </main>
      </div>

      {/* Toast Notifications */}
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#363636',
            color: '#fff',
          },
          success: {
            style: {
              background: '#22c55e',
            },
          },
          error: {
            style: {
              background: '#ef4444',
            },
          },
        }}
      />
    </div>
  );
};

export default Layout;
