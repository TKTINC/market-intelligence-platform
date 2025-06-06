import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import {
  TrendingUp,
  TrendingDown,
  DollarSign,
  Clock,
  AlertTriangle,
  CheckCircle,
} from 'lucide-react';
import { usePortfolioStore } from '@/store/portfolioStore';
import { useTradingStore } from '@/store/tradingStore';
import { useMarketDataStore } from '@/store/marketDataStore';
import { TradeRequest } from '@/types';
import Button from '@/components/ui/Button';
import Input from '@/components/ui/Input';
import LoadingSpinner from '@/components/ui/LoadingSpinner';
import toast from 'react-hot-toast';

const tradeSchema = z.object({
  symbol: z.string().min(1, 'Symbol is required').max(10, 'Symbol too long'),
  action: z.enum(['buy', 'sell', 'close']),
  quantity: z.number().min(1, 'Quantity must be at least 1').max(10000, 'Quantity too large'),
  order_type: z.enum(['market', 'limit', 'stop']),
  limit_price: z.number().optional(),
  stop_price: z.number().optional(),
  time_in_force: z.enum(['day', 'gtc', 'ioc']),
});

type TradeFormData = z.infer<typeof tradeSchema>;

const TradingInterface: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('');
  const { selectedPortfolio } = usePortfolioStore();
  const { executeTrade, isLoading: tradingLoading } = useTradingStore();
  const { quotes, subscribe, getQuote } = useMarketDataStore();

  const {
    register,
    handleSubmit,
    watch,
    setValue,
    formState: { errors },
    reset,
  } = useForm<TradeFormData>({
    resolver: zodResolver(tradeSchema),
    defaultValues: {
      action: 'buy',
      order_type: 'market',
      time_in_force: 'day',
      quantity: 1,
    },
  });

  const watchedAction = watch('action');
  const watchedOrderType = watch('order_type');
  const watchedQuantity = watch('quantity');

  useEffect(() => {
    if (selectedSymbol) {
      subscribe([selectedSymbol]);
    }
  }, [selectedSymbol, subscribe]);

  const currentQuote = selectedSymbol ? getQuote(selectedSymbol) : null;
  const currentPrice = currentQuote?.price || 0;

  const onSubmit = async (data: TradeFormData) => {
    if (!selectedPortfolio) {
      toast.error('Please select a portfolio first');
      return;
    }

    try {
      const tradeRequest: TradeRequest = {
        user_id: selectedPortfolio.user_id,
        portfolio_id: selectedPortfolio.portfolio_id,
        symbol: data.symbol.toUpperCase(),
        action: data.action,
        quantity: data.quantity,
        order_type: data.order_type,
        limit_price: data.limit_price,
        stop_price: data.stop_price,
        time_in_force: data.time_in_force,
      };

      await executeTrade(tradeRequest);
      reset();
      setSelectedSymbol('');
    } catch (error) {
      // Error is handled in the store
    }
  };

  const calculateTradeValue = () => {
    if (!currentPrice || !watchedQuantity) return 0;
    return currentPrice * watchedQuantity;
  };

  const getActionColor = (action: string) => {
    switch (action) {
      case 'buy':
        return 'text-success-600';
      case 'sell':
        return 'text-danger-600';
      case 'close':
        return 'text-warning-600';
      default:
        return 'text-gray-600';
    }
  };

  if (!selectedPortfolio) {
    return (
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center py-12">
          <AlertTriangle className="w-16 h-16 text-warning-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Portfolio Selected</h3>
          <p className="text-gray-600 mb-6">Please select a portfolio to start trading.</p>
          <Button variant="primary" onClick={() => window.location.href = '/portfolios'}>
            Go to Portfolios
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Trading Interface</h1>
        <p className="mt-1 text-sm text-gray-600">
          Execute virtual trades in your portfolio: {selectedPortfolio.name}
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Trading Form */}
        <div className="lg:col-span-2">
          <motion.div
            className="bg-white rounded-lg shadow-sm border border-gray-200 p-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <h2 className="text-lg font-semibold text-gray-900 mb-6">Place Order</h2>

            <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
              {/* Symbol and Quote */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Symbol
                  </label>
                  <Input
                    {...register('symbol')}
                    placeholder="e.g., AAPL"
                    value={selectedSymbol}
                    onChange={(value) => {
                      setSelectedSymbol(value.toUpperCase());
                      setValue('symbol', value.toUpperCase());
                    }}
                    error={errors.symbol?.message}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Current Price
                  </label>
                  <div className="flex items-center h-10 px-3 bg-gray-50 border border-gray-300 rounded-md">
                    {currentQuote ? (
                      <div className="flex items-center space-x-2">
                        <span className="font-medium">${currentPrice.toFixed(2)}</span>
                        <span className={`text-sm ${currentQuote.change >= 0 ? 'text-success-600' : 'text-danger-600'}`}>
                          {currentQuote.change >= 0 ? '+' : ''}{currentQuote.change_percent.toFixed(2)}%
                        </span>
                      </div>
                    ) : (
                      <span className="text-gray-500">Enter symbol</span>
                    )}
                  </div>
                </div>
              </div>

              {/* Action and Order Type */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Action
                  </label>
                  <select
                    {...register('action')}
                    className="block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
                  >
                    <option value="buy">Buy</option>
                    <option value="sell">Sell</option>
                    <option value="close">Close Position</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Order Type
                  </label>
                  <select
                    {...register('order_type')}
                    className="block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
                  >
                    <option value="market">Market</option>
                    <option value="limit">Limit</option>
                    <option value="stop">Stop</option>
                  </select>
                </div>
              </div>

              {/* Quantity and Prices */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Quantity
                  </label>
                  <Input
                    type="number"
                    {...register('quantity', { valueAsNumber: true })}
                    placeholder="100"
                    error={errors.quantity?.message}
                  />
                </div>

                {watchedOrderType === 'limit' && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Limit Price
                    </label>
                    <Input
                      type="number"
                      step="0.01"
                      {...register('limit_price', { valueAsNumber: true })}
                      placeholder="0.00"
                      error={errors.limit_price?.message}
                    />
                  </div>
                )}

                {watchedOrderType === 'stop' && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Stop Price
                    </label>
                    <Input
                      type="number"
                      step="0.01"
                      {...register('stop_price', { valueAsNumber: true })}
                      placeholder="0.00"
                      error={errors.stop_price?.message}
                    />
                  </div>
                )}

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Time in Force
                  </label>
                  <select
                    {...register('time_in_force')}
                    className="block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
                  >
                    <option value="day">Day</option>
                    <option value="gtc">Good Till Cancelled</option>
                    <option value="ioc">Immediate or Cancel</option>
                  </select>
                </div>
              </div>

              {/* Trade Summary */}
              <div className="bg-gray-50 rounded-lg p-4">
                <h3 className="text-sm font-medium text-gray-900 mb-3">Order Summary</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Action:</span>
                    <span className={`font-medium capitalize ${getActionColor(watchedAction)}`}>
                      {watchedAction}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Estimated Value:</span>
                    <span className="font-medium">
                      ${calculateTradeValue().toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Available Cash:</span>
                    <span className="font-medium">
                      ${selectedPortfolio.cash_balance.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </span>
                  </div>
                </div>
              </div>

              {/* Submit Button */}
              <div className="flex space-x-3">
                <Button
                  type="submit"
                  variant={watchedAction === 'buy' ? 'success' : watchedAction === 'sell' ? 'danger' : 'warning'}
                  loading={tradingLoading}
                  disabled={!selectedSymbol || !currentPrice}
                  className="flex-1"
                >
                  {watchedAction === 'buy' && <TrendingUp className="w-4 h-4 mr-2" />}
                  {watchedAction === 'sell' && <TrendingDown className="w-4 h-4 mr-2" />}
                  {watchedAction === 'close' && <Clock className="w-4 h-4 mr-2" />}
                  {watchedAction.charAt(0).toUpperCase() + watchedAction.slice(1)} {selectedSymbol || 'Stock'}
                </Button>

                <Button
                  type="button"
                  variant="secondary"
                  onClick={() => {
                    reset();
                    setSelectedSymbol('');
                  }}
                >
                  Clear
                </Button>
              </div>
            </form>
          </motion.div>
        </div>

        {/* Portfolio Summary */}
        <div className="space-y-6">
          {/* Portfolio Info */}
          <motion.div
            className="bg-white rounded-lg shadow-sm border border-gray-200 p-6"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
          >
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Portfolio Info</h3>
            
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">Total Value:</span>
                <span className="font-medium">
                  ${selectedPortfolio.total_value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-gray-600">Cash Balance:</span>
                <span className="font-medium">
                  ${selectedPortfolio.cash_balance.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-gray-600">Total P&L:</span>
                <span className={`font-medium ${selectedPortfolio.total_pnl >= 0 ? 'text-success-600' : 'text-danger-600'}`}>
                  {selectedPortfolio.total_pnl >= 0 ? '+' : ''}${selectedPortfolio.total_pnl.toFixed(2)}
                </span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-gray-600">Day P&L:</span>
                <span className={`font-medium ${selectedPortfolio.day_pnl >= 0 ? 'text-success-600' : 'text-danger-600'}`}>
                  {selectedPortfolio.day_pnl >= 0 ? '+' : ''}${selectedPortfolio.day_pnl.toFixed(2)}
                </span>
              </div>
              
              <div className="flex justify-between">
                <span className="text-gray-600">Positions:</span>
                <span className="font-medium">{selectedPortfolio.positions?.length || 0}</span>
              </div>
            </div>
          </motion.div>

          {/* Current Positions */}
          <motion.div
            className="bg-white rounded-lg shadow-sm border border-gray-200 p-6"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Current Positions</h3>
            
            {selectedPortfolio.positions && selectedPortfolio.positions.length > 0 ? (
              <div className="space-y-3">
                {selectedPortfolio.positions.slice(0, 5).map((position) => (
                  <div
                    key={position.position_id}
                    className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                  >
                    <div>
                      <p className="font-medium text-gray-900">{position.symbol}</p>
                      <p className="text-sm text-gray-600">{position.quantity} shares</p>
                    </div>
                    <div className="text-right">
                      <p className="font-medium">${position.market_value.toFixed(2)}</p>
                      <p className={`text-sm ${position.unrealized_pnl >= 0 ? 'text-success-600' : 'text-danger-600'}`}>
                        {position.unrealized_pnl >= 0 ? '+' : ''}${position.unrealized_pnl.toFixed(2)}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-6">
                <DollarSign className="w-12 h-12 text-gray-300 mx-auto mb-3" />
                <p className="text-gray-500">No positions</p>
                <p className="text-sm text-gray-400">Execute your first trade to get started</p>
              </div>
            )}
          </motion.div>

          {/* Risk Warning */}
          <motion.div
            className="bg-warning-50 border border-warning-200 rounded-lg p-4"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
          >
            <div className="flex items-start space-x-3">
              <AlertTriangle className="w-5 h-5 text-warning-600 flex-shrink-0 mt-0.5" />
              <div>
                <h4 className="text-sm font-medium text-warning-900">Virtual Trading</h4>
                <p className="text-sm text-warning-700 mt-1">
                  This is a virtual trading environment. No real money is involved, but all prices and market conditions are simulated to be realistic.
                </p>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default TradingInterface;
