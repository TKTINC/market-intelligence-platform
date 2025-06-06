import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import {
  Bot,
  Brain,
  Target,
  FileText,
  DollarSign,
  Clock,
  Zap,
  AlertCircle,
  TrendingUp,
} from 'lucide-react';
import { useAgentStore } from '@/store/agentStore';
import { useAuthStore } from '@/store/authStore';
import { AgentAnalysisRequest } from '@/types';
import Button from '@/components/ui/Button';
import Input from '@/components/ui/Input';
import LoadingSpinner from '@/components/ui/LoadingSpinner';
import toast from 'react-hot-toast';

const analysisSchema = z.object({
  symbols: z.string().min(1, 'At least one symbol is required'),
  agents: z.array(z.string()).min(1, 'Select at least one agent'),
  analysis_depth: z.enum(['quick', 'standard', 'comprehensive']),
  include_explanations: z.boolean(),
  max_cost_usd: z.number().min(0.1).max(10),
});

type AnalysisFormData = z.infer<typeof analysisSchema>;

interface AgentInfo {
  id: string;
  name: string;
  description: string;
  cost: number;
  icon: React.ComponentType<{ className?: string }>;
  capabilities: string[];
  processingTime: string;
}

const availableAgents: AgentInfo[] = [
  {
    id: 'sentiment',
    name: 'FinBERT Sentiment',
    description: 'Analyzes financial news and social sentiment using advanced NLP',
    cost: 0.02,
    icon: Brain,
    capabilities: ['News sentiment', 'Social media analysis', 'Market mood assessment'],
    processingTime: '5-10s',
  },
  {
    id: 'forecasting',
    name: 'TFT Forecasting',
    description: 'Predicts price movements using Temporal Fusion Transformers',
    cost: 0.08,
    icon: TrendingUp,
    capabilities: ['Price prediction', 'Volatility forecasting', 'Confidence intervals'],
    processingTime: '10-15s',
  },
  {
    id: 'strategy',
    name: 'GPT-4 Strategy',
    description: 'Generates sophisticated options trading strategies',
    cost: 0.45,
    icon: Target,
    capabilities: ['Options strategies', 'Risk assessment', 'Entry/exit conditions'],
    processingTime: '15-20s',
  },
  {
    id: 'explanation',
    name: 'Llama Explanation',
    description: 'Provides human-readable explanations of market analysis',
    cost: 0.15,
    icon: FileText,
    capabilities: ['Plain English explanations', 'Key insights', 'Risk factors'],
    processingTime: '20-30s',
  },
];

const tierLimits = {
  free: { maxCost: 0.25, allowedAgents: ['sentiment'] },
  basic: { maxCost: 1.0, allowedAgents: ['sentiment', 'forecasting'] },
  premium: { maxCost: 2.5, allowedAgents: ['sentiment', 'forecasting', 'strategy'] },
  enterprise: { maxCost: 5.0, allowedAgents: ['sentiment', 'forecasting', 'strategy', 'explanation'] },
};

const AgentSelection: React.FC = () => {
  const [selectedAgents, setSelectedAgents] = useState<string[]>(['sentiment']);
  const { requestAnalysis, isLoading, currentAnalysis } = useAgentStore();
  const { user } = useAuthStore();

  const {
    register,
    handleSubmit,
    watch,
    setValue,
    formState: { errors },
  } = useForm<AnalysisFormData>({
    resolver: zodResolver(analysisSchema),
    defaultValues: {
      agents: ['sentiment'],
      analysis_depth: 'standard',
      include_explanations: true,
      max_cost_usd: 1.0,
    },
  });

  const watchedDepth = watch('analysis_depth');
  const watchedCost = watch('max_cost_usd');

  const userTier = user?.tier || 'free';
  const tierLimit = tierLimits[userTier];

  const toggleAgent = (agentId: string) => {
    if (!tierLimit.allowedAgents.includes(agentId)) {
      toast.error(`${agentId} agent requires ${agentId === 'strategy' ? 'Premium' : 'Enterprise'} tier`);
      return;
    }

    const newSelectedAgents = selectedAgents.includes(agentId)
      ? selectedAgents.filter(id => id !== agentId)
      : [...selectedAgents, agentId];

    setSelectedAgents(newSelectedAgents);
    setValue('agents', newSelectedAgents);
  };

  const calculateTotalCost = () => {
    const baseCost = selectedAgents.reduce((sum, agentId) => {
      const agent = availableAgents.find(a => a.id === agentId);
      return sum + (agent?.cost || 0);
    }, 0);

    const depthMultiplier = {
      quick: 0.7,
      standard: 1.0,
      comprehensive: 1.5,
    }[watchedDepth];

    return baseCost * depthMultiplier;
  };

  const totalCost = calculateTotalCost();

  const onSubmit = async (data: AnalysisFormData) => {
    try {
      const symbols = data.symbols
        .split(',')
        .map(s => s.trim().toUpperCase())
        .filter(s => s.length > 0);

      if (symbols.length === 0) {
        toast.error('Please enter at least one symbol');
        return;
      }

      if (totalCost > tierLimit.maxCost) {
        toast.error(`Cost exceeds ${userTier} tier limit of ${tierLimit.maxCost}`);
        return;
      }

      const request: AgentAnalysisRequest = {
        user_id: user!.user_id,
        symbols,
        agents: selectedAgents as any[],
        analysis_depth: data.analysis_depth,
        include_explanations: data.include_explanations,
        max_cost_usd: data.max_cost_usd,
      };

      await requestAnalysis(request);
    } catch (error) {
      // Error is handled in the store
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">AI Analysis</h1>
        <p className="mt-1 text-sm text-gray-600">
          Select AI agents to analyze market data and generate insights for your trading decisions.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Analysis Form */}
        <div className="lg:col-span-2 space-y-6">
          {/* Agent Selection */}
          <motion.div
            className="bg-white rounded-lg shadow-sm border border-gray-200 p-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <h2 className="text-lg font-semibold text-gray-900 mb-6">Select AI Agents</h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {availableAgents.map((agent) => {
                const isSelected = selectedAgents.includes(agent.id);
                const isAllowed = tierLimit.allowedAgents.includes(agent.id);

                return (
                  <motion.div
                    key={agent.id}
                    className={`relative p-4 rounded-lg border-2 cursor-pointer transition-all ${
                      isSelected
                        ? 'border-primary-500 bg-primary-50'
                        : isAllowed
                        ? 'border-gray-200 hover:border-gray-300'
                        : 'border-gray-100 bg-gray-50 cursor-not-allowed opacity-60'
                    }`}
                    onClick={() => isAllowed && toggleAgent(agent.id)}
                    whileHover={isAllowed ? { scale: 1.02 } : {}}
                    whileTap={isAllowed ? { scale: 0.98 } : {}}
                  >
                    <div className="flex items-start space-x-3">
                      <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                        isSelected ? 'bg-primary-600 text-white' : 'bg-gray-100 text-gray-600'
                      }`}>
                        <agent.icon className="w-5 h-5" />
                      </div>
                      
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between">
                          <h3 className="text-sm font-medium text-gray-900">{agent.name}</h3>
                          <span className="text-sm font-medium text-gray-500">${agent.cost}</span>
                        </div>
                        
                        <p className="text-xs text-gray-600 mt-1">{agent.description}</p>
                        
                        <div className="mt-2">
                          <div className="flex items-center space-x-1 text-xs text-gray-500 mb-1">
                            <Clock className="w-3 h-3" />
                            <span>{agent.processingTime}</span>
                          </div>
                          
                          <div className="flex flex-wrap gap-1">
                            {agent.capabilities.slice(0, 2).map((capability) => (
                              <span
                                key={capability}
                                className="px-2 py-0.5 text-xs bg-gray-100 text-gray-600 rounded"
                              >
                                {capability}
                              </span>
                            ))}
                            {agent.capabilities.length > 2 && (
                              <span className="px-2 py-0.5 text-xs bg-gray-100 text-gray-600 rounded">
                                +{agent.capabilities.length - 2} more
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>

                    {!isAllowed && (
                      <div className="absolute top-2 right-2">
                        <div className="w-6 h-6 bg-warning-100 rounded-full flex items-center justify-center">
                          <AlertCircle className="w-4 h-4 text-warning-600" />
                        </div>
                      </div>
                    )}
                  </motion.div>
                );
              })}
            </div>
          </motion.div>

          {/* Analysis Configuration */}
          <motion.div
            className="bg-white rounded-lg shadow-sm border border-gray-200 p-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <h2 className="text-lg font-semibold text-gray-900 mb-6">Analysis Configuration</h2>

            <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
              {/* Symbols */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Symbols (comma-separated)
                </label>
                <Input
                  {...register('symbols')}
                  placeholder="AAPL, MSFT, GOOGL"
                  error={errors.symbols?.message}
                />
                <p className="text-xs text-gray-500 mt-1">
                  Enter stock symbols separated by commas (max 10 symbols)
                </p>
              </div>

              {/* Analysis Depth */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Analysis Depth
                </label>
                <select
                  {...register('analysis_depth')}
                  className="block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
                >
                  <option value="quick">Quick (0.7x cost)</option>
                  <option value="standard">Standard (1.0x cost)</option>
                  <option value="comprehensive">Comprehensive (1.5x cost)</option>
                </select>
              </div>

              {/* Options */}
              <div>
                <div className="flex items-center">
                  <input
                    {...register('include_explanations')}
                    type="checkbox"
                    className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
                  />
                  <label className="ml-2 block text-sm text-gray-900">
                    Include detailed explanations
                  </label>
                </div>
              </div>

              {/* Max Cost */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Maximum Cost (USD)
                </label>
                <Input
                  type="number"
                  step="0.01"
                  {...register('max_cost_usd', { valueAsNumber: true })}
                  placeholder="1.00"
                  error={errors.max_cost_usd?.message}
                />
                <p className="text-xs text-gray-500 mt-1">
                  Your {userTier} tier limit: ${tierLimit.maxCost}
                </p>
              </div>

              {/* Submit Button */}
              <Button
                type="submit"
                variant="primary"
                loading={isLoading}
                disabled={selectedAgents.length === 0 || totalCost > tierLimit.maxCost}
                className="w-full"
              >
                <Bot className="w-4 h-4 mr-2" />
                Start Analysis (${totalCost.toFixed(2)})
              </Button>
            </form>
          </motion.div>
        </div>

        {/* Analysis Summary */}
        <div className="space-y-6">
          {/* Cost Summary */}
          <motion.div
            className="bg-white rounded-lg shadow-sm border border-gray-200 p-6"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Cost Summary</h3>
            
            <div className="space-y-3">
              {selectedAgents.map((agentId) => {
                const agent = availableAgents.find(a => a.id === agentId);
                if (!agent) return null;

                return (
                  <div key={agentId} className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">{agent.name}</span>
                    <span className="text-sm font-medium">${agent.cost.toFixed(2)}</span>
                  </div>
                );
              })}
              
              <div className="pt-3 border-t border-gray-200">
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium text-gray-900">
                    Depth Multiplier ({watchedDepth})
                  </span>
                  <span className="text-sm font-medium">
                    {watchedDepth === 'quick' ? '0.7x' : watchedDepth === 'standard' ? '1.0x' : '1.5x'}
                  </span>
                </div>
              </div>
              
              <div className="pt-3 border-t border-gray-200">
                <div className="flex justify-between items-center">
                  <span className="font-medium text-gray-900">Total Cost</span>
                  <span className={`font-bold ${totalCost > tierLimit.maxCost ? 'text-danger-600' : 'text-primary-600'}`}>
                    ${totalCost.toFixed(2)}
                  </span>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Tier Info */}
          <motion.div
            className="bg-primary-50 border border-primary-200 rounded-lg p-4"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
          >
            <div className="flex items-start space-x-3">
              <DollarSign className="w-5 h-5 text-primary-600 flex-shrink-0 mt-0.5" />
              <div>
                <h4 className="text-sm font-medium text-primary-900">Your {userTier} Tier</h4>
                <p className="text-sm text-primary-700 mt-1">
                  Maximum cost per analysis: ${tierLimit.maxCost}
                </p>
                <p className="text-sm text-primary-700">
                  Available agents: {tierLimit.allowedAgents.length} of {availableAgents.length}
                </p>
              </div>
            </div>
          </motion.div>

          {/* Current Analysis */}
          {currentAnalysis && (
            <motion.div
              className="bg-white rounded-lg shadow-sm border border-gray-200 p-6"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
            >
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Latest Analysis</h3>
              
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Symbols:</span>
                  <span className="font-medium">{currentAnalysis.symbols.join(', ')}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Agents Used:</span>
                  <span className="font-medium">{currentAnalysis.agents_used.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Cost:</span>
                  <span className="font-medium">${currentAnalysis.total_cost_usd.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Processing Time:</span>
                  <span className="font-medium">{currentAnalysis.processing_time_ms}ms</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Confidence:</span>
                  <span className="font-medium">{(currentAnalysis.overall_confidence * 100).toFixed(1)}%</span>
                </div>
              </div>
              
              <div className="mt-4 pt-4 border-t border-gray-200">
                <Button variant="secondary" size="sm" className="w-full">
                  View Full Results
                </Button>
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AgentSelection;
