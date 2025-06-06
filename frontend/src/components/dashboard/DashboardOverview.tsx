import React, { useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  TrendingUp,
  TrendingDown,
  DollarSign,
  Briefcase,
  Activity,
  Users,
} from 'lucide-react';
import { usePortfolioStore } from '@/store/portfolioStore';
import { useMarketDataStore } from '@/store/marketDataStore';
import LoadingSpinner from '@/components/ui/LoadingSpinner';

interface StatCardProps {
  title: string;
  value: string;
  change?: string;
  changeType?: 'positive' | 'negative' | 'neutral';
  icon: React.ComponentType<{ className?: string }>;
}

const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  change,
  changeType = 'neutral',
  icon: Icon,
}) => {
  const changeColors = {
    positive: 'text-success-600 bg-success-50',
    negative: 'text-danger-600 bg-danger-50',
    neutral: 'text-gray-600 bg-gray-50',
  };

  return (
    <motion.div
      className="bg-white rounded-lg shadow-sm border border-gray-200 p-6"
      whileHover={{ y: -2 }}
      transition={{ duration: 0.2 }}
    >
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-bold text-gray-900 mt-1">{value}</p>
          {change && (
            <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium mt-2 ${changeColors[changeType]}`}>
              {changeType === 'positive' && <TrendingUp className="w-3 h-3 mr-1" />}
              {changeType === 'negative' && <TrendingDown className="w-3 h-3 mr-1" />}
              {change}
            </div>
          )}
        </div>
        <div className="w-12 h-12 bg-primary-50 rounded-lg flex items-center justify-center">
          <Icon className="w-6 h-6 text-primary-600" />
        </div>
      </div>
    </motion.div>
  );
};

const DashboardOverview: React.FC = () => {
  const { portfolios, fetchPortfolios, isLoading } = usePortfolioStore();
  const { quotes } = useMarketDataStore();

  useEffect(() => {
    fetchPortfolios();
  }, [fetchPortfolios]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner size="lg" text="Loading dashboard..." />
      </div>
    );
  }

  // Calculate aggregated statistics
  const totalValue = portfolios.reduce((sum, p) => sum + p.total_value, 0);
  const totalPnL = portfolios.reduce((sum, p) => sum + p.total_pnl, 0);
  const dayPnL = portfolios.reduce((sum, p) => sum + p.day_pnl, 0);
  const totalPositions = portfolios.reduce((sum, p) => sum + (p.positions?.length || 0), 0);

  const pnlChangeType = totalPnL >= 0 ? 'positive' : 'negative';
  const dayPnLChangeType = dayPnL >= 0 ? 'positive' : 'negative';

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="mt-1 text-sm text-gray-600">
          Welcome back! Here's an overview of your market intelligence platform.
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard
          title="Total Portfolio Value"
          value={`${totalValue.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
          change={`${totalPnL >= 0 ? '+' : ''}${Math.abs(totalPnL).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
          changeType={pnlChangeType}
          icon={DollarSign}
        />
        
        <StatCard
          title="Day P&L"
          value={`${dayPnL >= 0 ? '+' : '-'}${Math.abs(dayPnL).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
          change={`${((dayPnL / totalValue) * 100).toFixed(2)}%`}
          changeType={dayPnLChangeType}
          icon={Activity}
        />
        
        <StatCard
          title="Active Portfolios"
          value={portfolios.length.toString()}
          icon={Briefcase}
        />
        
        <StatCard
          title="Total Positions"
          value={totalPositions.toString()}
          icon={Users}
        />
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Portfolios */}
        <motion.div
          className="bg-white rounded-lg shadow-sm border border-gray-200"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <div className="p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Portfolios</h3>
            
            {portfolios.length === 0 ? (
              <div className="text-center py-8">
                <Briefcase className="w-12 h-12 text-gray-300 mx-auto mb-4" />
                <p className="text-gray-500">No portfolios found</p>
                <p className="text-sm text-gray-400 mt-1">Create your first portfolio to get started</p>
              </div>
            ) : (
              <div className="space-y-3">
                {portfolios.slice(0, 3).map((portfolio) => (
                  <div
                    key={portfolio.portfolio_id}
                    className="flex items-center justify-between p-3 rounded-lg border border-gray-100 hover:bg-gray-50 transition-colors"
                  >
                    <div className="flex items-center space-x-3">
                      <div className="w-10 h-10 bg-primary-100 rounded-lg flex items-center justify-center">
                        <Briefcase className="w-5 h-5 text-primary-600" />
                      </div>
                      <div>
                        <p className="font-medium text-gray-900">{portfolio.name}</p>
                        <p className="text-sm text-gray-500">
                          {portfolio.positions?.length || 0} positions
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="font-medium text-gray-900">
                        ${portfolio.total_value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </p>
                      <p className={`text-sm ${portfolio.total_pnl >= 0 ? 'text-success-600' : 'text-danger-600'}`}>
                        {portfolio.total_pnl >= 0 ? '+' : ''}${portfolio.total_pnl.toFixed(2)}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </motion.div>

        {/* Market Overview */}
        <motion.div
          className="bg-white rounded-lg shadow-sm border border-gray-200"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Market Overview</h3>
            
            {Object.keys(quotes).length === 0 ? (
              <div className="text-center py-8">
                <Activity className="w-12 h-12 text-gray-300 mx-auto mb-4" />
                <p className="text-gray-500">No market data available</p>
                <p className="text-sm text-gray-400 mt-1">Market data will appear when you start trading</p>
              </div>
            ) : (
              <div className="space-y-3">
                {Object.values(quotes).slice(0, 4).map((quote) => (
                  <div
                    key={quote.symbol}
                    className="flex items-center justify-between p-3 rounded-lg border border-gray-100"
                  >
                    <div className="font-medium text-gray-900">{quote.symbol}</div>
                    <div className="text-right">
                      <p className="font-medium text-gray-900">${quote.price.toFixed(2)}</p>
                      <p className={`text-sm ${quote.change >= 0 ? 'text-success-600' : 'text-danger-600'}`}>
                        {quote.change >= 0 ? '+' : ''}{quote.change_percent.toFixed(2)}%
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default DashboardOverview;
