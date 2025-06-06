import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import {
  Plus,
  Briefcase,
  DollarSign,
  TrendingUp,
  TrendingDown,
  Settings,
  Trash2,
  Eye,
  MoreVertical,
  PieChart,
  BarChart3,
} from 'lucide-react';
import { usePortfolioStore } from '@/store/portfolioStore';
import { Portfolio } from '@/types';
import Button from '@/components/ui/Button';
import Input from '@/components/ui/Input';
import Modal from '@/components/ui/Modal';
import LoadingSpinner from '@/components/ui/LoadingSpinner';
import { formatCurrency, formatPercentage } from '@/utils/formatters';

const portfolioSchema = z.object({
  name: z.string().min(1, 'Portfolio name is required').max(50, 'Name too long'),
  initial_balance: z.number().min(1000, 'Minimum balance is $1,000').max(1000000, 'Maximum balance is $1,000,000'),
  risk_tolerance: z.enum(['low', 'medium', 'high']),
  max_position_size: z.number().min(0.01, 'Minimum 1%').max(0.5, 'Maximum 50%'),
});

type PortfolioFormData = z.infer<typeof portfolioSchema>;

interface PortfolioCardProps {
  portfolio: Portfolio;
  onSelect: (portfolio: Portfolio) => void;
  onEdit: (portfolio: Portfolio) => void;
  onDelete: (portfolio: Portfolio) => void;
  isSelected: boolean;
}

const PortfolioCard: React.FC<PortfolioCardProps> = ({
  portfolio,
  onSelect,
  onEdit,
  onDelete,
  isSelected,
}) => {
  const [showMenu, setShowMenu] = useState(false);

  const pnlColor = portfolio.total_pnl >= 0 ? 'text-success-600' : 'text-danger-600';
  const dayPnlColor = portfolio.day_pnl >= 0 ? 'text-success-600' : 'text-danger-600';

  return (
    <motion.div
      className={`relative bg-white rounded-lg border-2 p-6 cursor-pointer transition-all ${
        isSelected ? 'border-primary-500 shadow-md' : 'border-gray-200 hover:border-gray-300'
      }`}
      whileHover={{ y: -2 }}
      onClick={() => onSelect(portfolio)}
      layout
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${
            isSelected ? 'bg-primary-100' : 'bg-gray-100'
          }`}>
            <Briefcase className={`w-6 h-6 ${
              isSelected ? 'text-primary-600' : 'text-gray-600'
            }`} />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-900">{portfolio.name}</h3>
            <p className="text-sm text-gray-500 capitalize">
              {portfolio.risk_tolerance} risk • {portfolio.positions?.length || 0} positions
            </p>
          </div>
        </div>

        <div className="relative">
          <button
            onClick={(e) => {
              e.stopPropagation();
              setShowMenu(!showMenu);
            }}
            className="p-1 text-gray-400 hover:text-gray-600 transition-colors"
          >
            <MoreVertical className="w-5 h-5" />
          </button>

          <AnimatePresence>
            {showMenu && (
              <motion.div
                className="absolute right-0 top-8 w-48 bg-white rounded-md shadow-lg border border-gray-200 py-1 z-10"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ duration: 0.1 }}
              >
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onEdit(portfolio);
                    setShowMenu(false);
                  }}
                  className="flex items-center w-full px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                >
                  <Settings className="w-4 h-4 mr-2" />
                  Edit Portfolio
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete(portfolio);
                    setShowMenu(false);
                  }}
                  className="flex items-center w-full px-4 py-2 text-sm text-danger-600 hover:bg-gray-100"
                >
                  <Trash2 className="w-4 h-4 mr-2" />
                  Delete Portfolio
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <p className="text-sm text-gray-600">Total Value</p>
          <p className="text-xl font-bold text-gray-900">
            {formatCurrency(portfolio.total_value)}
          </p>
        </div>
        <div>
          <p className="text-sm text-gray-600">Cash Balance</p>
          <p className="text-lg font-semibold text-gray-700">
            {formatCurrency(portfolio.cash_balance)}
          </p>
        </div>
      </div>

      {/* P&L */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-sm text-gray-600">Total P&L</p>
          <div className="flex items-center space-x-1">
            {portfolio.total_pnl >= 0 ? (
              <TrendingUp className="w-4 h-4 text-success-600" />
            ) : (
              <TrendingDown className="w-4 h-4 text-danger-600" />
            )}
            <p className={`font-semibold ${pnlColor}`}>
              {formatCurrency(portfolio.total_pnl, true)}
            </p>
          </div>
        </div>
        <div>
          <p className="text-sm text-gray-600">Day P&L</p>
          <div className="flex items-center space-x-1">
            {portfolio.day_pnl >= 0 ? (
              <TrendingUp className="w-4 h-4 text-success-600" />
            ) : (
              <TrendingDown className="w-4 h-4 text-danger-600" />
            )}
            <p className={`font-semibold ${dayPnlColor}`}>
              {formatCurrency(portfolio.day_pnl, true)}
            </p>
          </div>
        </div>
      </div>

      {/* Risk Metrics */}
      {portfolio.risk_metrics && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600">Concentration Risk</span>
            <span className={`font-medium ${
              portfolio.risk_metrics.concentration_risk > 0.3 ? 'text-warning-600' : 'text-success-600'
            }`}>
              {formatPercentage(portfolio.risk_metrics.concentration_risk)}
            </span>
          </div>
        </div>
      )}

      {/* Selection Indicator */}
      {isSelected && (
        <div className="absolute top-3 right-3">
          <div className="w-3 h-3 bg-primary-600 rounded-full"></div>
        </div>
      )}
    </motion.div>
  );
};

const PortfolioManager: React.FC = () => {
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [editingPortfolio, setEditingPortfolio] = useState<Portfolio | null>(null);
  const [deletingPortfolio, setDeletingPortfolio] = useState<Portfolio | null>(null);

  const {
    portfolios,
    selectedPortfolio,
    isLoading,
    fetchPortfolios,
    createPortfolio,
    updatePortfolio,
    deletePortfolio,
    selectPortfolio,
  } = usePortfolioStore();

  const {
    register,
    handleSubmit,
    reset,
    setValue,
    formState: { errors },
  } = useForm<PortfolioFormData>({
    resolver: zodResolver(portfolioSchema),
    defaultValues: {
      initial_balance: 100000,
      risk_tolerance: 'medium',
      max_position_size: 0.1,
    },
  });

  useEffect(() => {
    fetchPortfolios();
  }, [fetchPortfolios]);

  useEffect(() => {
    if (editingPortfolio) {
      setValue('name', editingPortfolio.name);
      setValue('initial_balance', editingPortfolio.initial_balance);
      setValue('risk_tolerance', editingPortfolio.risk_tolerance);
      setValue('max_position_size', editingPortfolio.max_position_size);
    }
  }, [editingPortfolio, setValue]);

  const onSubmit = async (data: PortfolioFormData) => {
    try {
      if (editingPortfolio) {
        await updatePortfolio(editingPortfolio.portfolio_id, data);
        setEditingPortfolio(null);
      } else {
        await createPortfolio(data);
        setShowCreateModal(false);
      }
      reset();
    } catch (error) {
      // Error is handled in the store
    }
  };

  const handleDelete = async () => {
    if (deletingPortfolio) {
      try {
        await deletePortfolio(deletingPortfolio.portfolio_id);
        setDeletingPortfolio(null);
      } catch (error) {
        // Error is handled in the store
      }
    }
  };

  const handleSelect = (portfolio: Portfolio) => {
    selectPortfolio(portfolio.portfolio_id);
  };

  if (isLoading && portfolios.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner size="lg" text="Loading portfolios..." />
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Portfolio Manager</h1>
          <p className="mt-1 text-sm text-gray-600">
            Create and manage your virtual trading portfolios
          </p>
        </div>
        <Button
          variant="primary"
          onClick={() => setShowCreateModal(true)}
          className="flex items-center space-x-2"
        >
          <Plus className="w-4 h-4" />
          <span>New Portfolio</span>
        </Button>
      </div>

      {/* Portfolio Grid */}
      {portfolios.length === 0 ? (
        <motion.div
          className="text-center py-12"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <Briefcase className="w-16 h-16 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Portfolios Yet</h3>
          <p className="text-gray-600 mb-6">
            Create your first portfolio to start virtual trading with AI-powered market intelligence.
          </p>
          <Button
            variant="primary"
            onClick={() => setShowCreateModal(true)}
            className="flex items-center space-x-2"
          >
            <Plus className="w-4 h-4" />
            <span>Create Portfolio</span>
          </Button>
        </motion.div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {portfolios.map((portfolio) => (
            <PortfolioCard
              key={portfolio.portfolio_id}
              portfolio={portfolio}
              onSelect={handleSelect}
              onEdit={setEditingPortfolio}
              onDelete={setDeletingPortfolio}
              isSelected={selectedPortfolio?.portfolio_id === portfolio.portfolio_id}
            />
          ))}
        </div>
      )}

      {/* Portfolio Summary */}
      {selectedPortfolio && (
        <motion.div
          className="mt-8 bg-white rounded-lg shadow-sm border border-gray-200 p-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          key={selectedPortfolio.portfolio_id}
        >
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold text-gray-900">
              {selectedPortfolio.name} - Portfolio Details
            </h2>
            <div className="flex space-x-2">
              <Button variant="secondary" size="sm">
                <BarChart3 className="w-4 h-4 mr-2" />
                Analytics
              </Button>
              <Button variant="secondary" size="sm">
                <PieChart className="w-4 h-4 mr-2" />
                Allocation
              </Button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
            <div className="bg-gray-50 rounded-lg p-4">
              <p className="text-sm text-gray-600 mb-1">Total Value</p>
              <p className="text-2xl font-bold text-gray-900">
                {formatCurrency(selectedPortfolio.total_value)}
              </p>
            </div>
            <div className="bg-gray-50 rounded-lg p-4">
              <p className="text-sm text-gray-600 mb-1">Cash Balance</p>
              <p className="text-2xl font-bold text-gray-900">
                {formatCurrency(selectedPortfolio.cash_balance)}
              </p>
            </div>
            <div className="bg-gray-50 rounded-lg p-4">
              <p className="text-sm text-gray-600 mb-1">Total P&L</p>
              <p className={`text-2xl font-bold ${selectedPortfolio.total_pnl >= 0 ? 'text-success-600' : 'text-danger-600'}`}>
                {formatCurrency(selectedPortfolio.total_pnl, true)}
              </p>
            </div>
            <div className="bg-gray-50 rounded-lg p-4">
              <p className="text-sm text-gray-600 mb-1">Day P&L</p>
              <p className={`text-2xl font-bold ${selectedPortfolio.day_pnl >= 0 ? 'text-success-600' : 'text-danger-600'}`}>
                {formatCurrency(selectedPortfolio.day_pnl, true)}
              </p>
            </div>
          </div>

          {/* Positions */}
          {selectedPortfolio.positions && selectedPortfolio.positions.length > 0 ? (
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-4">Current Positions</h3>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Symbol
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Quantity
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Avg Cost
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Current Price
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Market Value
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        P&L
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {selectedPortfolio.positions.map((position) => (
                      <tr key={position.position_id} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {position.symbol}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {position.quantity.toLocaleString()}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {formatCurrency(position.avg_cost)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {formatCurrency(position.current_price)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {formatCurrency(position.market_value)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm">
                          <div className={`font-medium ${position.unrealized_pnl >= 0 ? 'text-success-600' : 'text-danger-600'}`}>
                            {formatCurrency(position.unrealized_pnl, true)}
                          </div>
                          <div className={`text-xs ${position.pnl_percentage >= 0 ? 'text-success-600' : 'text-danger-600'}`}>
                            {formatPercentage(position.pnl_percentage / 100, true)}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="text-center py-8">
              <DollarSign className="w-12 h-12 text-gray-300 mx-auto mb-3" />
              <p className="text-gray-500">No positions in this portfolio</p>
              <p className="text-sm text-gray-400">Start trading to see your positions here</p>
            </div>
          )}
        </motion.div>
      )}

      {/* Create/Edit Portfolio Modal */}
      <Modal
        isOpen={showCreateModal || !!editingPortfolio}
        onClose={() => {
          setShowCreateModal(false);
          setEditingPortfolio(null);
          reset();
        }}
        title={editingPortfolio ? 'Edit Portfolio' : 'Create New Portfolio'}
        size="md"
      >
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Portfolio Name
            </label>
            <Input
              {...register('name')}
              placeholder="My Trading Portfolio"
              error={errors.name?.message}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Initial Balance
            </label>
            <Input
              type="number"
              {...register('initial_balance', { valueAsNumber: true })}
              placeholder="100000"
              error={errors.initial_balance?.message}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Risk Tolerance
            </label>
            <select
              {...register('risk_tolerance')}
              className="block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
            >
              <option value="low">Low Risk</option>
              <option value="medium">Medium Risk</option>
              <option value="high">High Risk</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Maximum Position Size (% of Portfolio)
            </label>
            <Input
              type="number"
              step="0.01"
              {...register('max_position_size', { valueAsNumber: true })}
              placeholder="0.10"
              error={errors.max_position_size?.message}
            />
            <p className="text-xs text-gray-500 mt-1">
              Maximum percentage of portfolio value for a single position
            </p>
          </div>

          <div className="flex space-x-3 pt-4">
            <Button
              type="submit"
              variant="primary"
              loading={isLoading}
              className="flex-1"
            >
              {editingPortfolio ? 'Update Portfolio' : 'Create Portfolio'}
            </Button>
            <Button
              type="button"
              variant="secondary"
              onClick={() => {
                setShowCreateModal(false);
                setEditingPortfolio(null);
                reset();
              }}
            >
              Cancel
            </Button>
          </div>
        </form>
      </Modal>

      {/* Delete Confirmation Modal */}
      <Modal
        isOpen={!!deletingPortfolio}
        onClose={() => setDeletingPortfolio(null)}
        title="Delete Portfolio"
        size="sm"
      >
        <div className="text-center">
          <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-danger-100 mb-4">
            <Trash2 className="h-6 w-6 text-danger-600" />
          </div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            Delete {deletingPortfolio?.name}?
          </h3>
          <p className="text-sm text-gray-500 mb-6">
            This action cannot be undone. All positions and trading history will be permanently deleted.
          </p>
          <div className="flex space-x-3">
            <Button
              variant="danger"
              onClick={handleDelete}
              loading={isLoading}
              className="flex-1"
            >
              Delete Portfolio
            </Button>
            <Button
              variant="secondary"
              onClick={() => setDeletingPortfolio(null)}
              className="flex-1"
            >
              Cancel
            </Button>
          </div>
        </div>
      </Modal>
    </div>
  );
};

export default PortfolioManager;
