// Core API Types
export interface User {
  user_id: string;
  email: string;
  username: string;
  role: 'user' | 'premium' | 'admin';
  tier: 'free' | 'basic' | 'premium' | 'enterprise';
  permissions: string[];
  created_at: string;
  last_login: string;
  is_active: boolean;
}

export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
  session_id: string;
  user: User;
}

export interface LoginRequest {
  email: string;
  password: string;
}

// Portfolio Types
export interface Portfolio {
  portfolio_id: string;
  user_id: string;
  name: string;
  initial_balance: number;
  cash_balance: number;
  total_value: number;
  total_pnl: number;
  day_pnl: number;
  risk_tolerance: 'low' | 'medium' | 'high';
  max_position_size: number;
  positions: Position[];
  risk_metrics: RiskMetrics;
  created_at: string;
  last_updated: string;
  status: 'active' | 'inactive';
}

export interface Position {
  position_id: string;
  portfolio_id: string;
  symbol: string;
  quantity: number;
  avg_cost: number;
  current_price: number;
  market_value: number;
  unrealized_pnl: number;
  realized_pnl: number;
  day_pnl: number;
  pnl_percentage: number;
  opened_at: string;
  last_updated: string;
}

export interface RiskMetrics {
  var_1d: number;
  var_5d: number;
  max_drawdown: number;
  concentration_risk: number;
  leverage_ratio: number;
  sharpe_ratio: number;
  beta: number;
  correlation_risk: number;
  liquidity_risk: number;
  position_count: number;
  largest_position_pct: number;
  cash_ratio: number;
}

// Trading Types
export interface TradeRequest {
  user_id: string;
  portfolio_id: string;
  symbol: string;
  action: 'buy' | 'sell' | 'close';
  quantity: number;
  order_type: 'market' | 'limit' | 'stop';
  limit_price?: number;
  stop_price?: number;
  time_in_force: 'day' | 'gtc' | 'ioc';
}

export interface TradeResponse {
  trade_id: string;
  user_id: string;
  portfolio_id: string;
  symbol: string;
  action: string;
  quantity: number;
  executed_price: number;
  total_value: number;
  commission: number;
  timestamp: string;
  status: 'executed' | 'pending' | 'cancelled';
}

export interface TradeHistory {
  trades: TradeResponse[];
  total_count: number;
}

// Agent Analysis Types
export interface AgentAnalysisRequest {
  user_id: string;
  symbols: string[];
  agents: ('sentiment' | 'forecasting' | 'strategy' | 'explanation')[];
  analysis_depth: 'quick' | 'standard' | 'comprehensive';
  include_explanations: boolean;
  max_cost_usd: number;
}

export interface AgentAnalysisResponse {
  request_id: string;
  symbols: string[];
  agents_used: string[];
  processing_time_ms: number;
  total_cost_usd: number;
  sentiment_analysis?: SentimentAnalysis;
  price_forecasts?: PriceForecasts;
  strategy_recommendations?: StrategyRecommendations;
  explanations?: Explanations;
  overall_confidence: number;
  timestamp: string;
}

export interface SentimentAnalysis {
  [symbol: string]: {
    sentiment_score: number;
    sentiment_label: 'positive' | 'negative' | 'neutral';
    confidence: number;
    news_impact: number;
    social_sentiment: number;
    analyst_sentiment: number;
  };
}

export interface PriceForecasts {
  [symbol: string]: {
    horizons: {
      [days: string]: {
        predicted_price: number;
        confidence_interval: [number, number];
        probability_up: number;
        volatility_forecast: number;
      };
    };
    trend_direction: 'bullish' | 'bearish' | 'neutral';
    key_levels: {
      support: number[];
      resistance: number[];
    };
  };
}

export interface StrategyRecommendations {
  [symbol: string]: {
    recommended_strategies: OptionsStrategy[];
    risk_assessment: string;
    market_outlook: string;
    entry_conditions: string[];
    exit_conditions: string[];
  };
}

export interface OptionsStrategy {
  strategy_name: string;
  strategy_type: 'bullish' | 'bearish' | 'neutral' | 'volatility';
  legs: OptionsLeg[];
  max_profit: number;
  max_loss: number;
  break_even_points: number[];
  probability_profit: number;
  capital_required: number;
  greeks: {
    delta: number;
    gamma: number;
    theta: number;
    vega: number;
  };
}

export interface OptionsLeg {
  action: 'buy' | 'sell';
  option_type: 'call' | 'put';
  strike: number;
  expiry: string;
  quantity: number;
  premium: number;
}

export interface Explanations {
  [symbol: string]: {
    market_summary: string;
    key_insights: string[];
    risk_factors: string[];
    opportunities: string[];
    technical_analysis: string;
    fundamental_analysis: string;
  };
}

// Market Data Types
export interface MarketQuote {
  symbol: string;
  price: number;
  bid: number;
  ask: number;
  volume: number;
  change: number;
  change_percent: number;
  high: number;
  low: number;
  open_price: number;
  timestamp: string;
}

export interface OptionsData {
  symbol: string;
  option_type: 'call' | 'put';
  strike: number;
  expiry: string;
  bid: number;
  ask: number;
  last: number;
  volume: number;
  open_interest: number;
  implied_volatility: number;
  delta: number;
  gamma: number;
  theta: number;
  vega: number;
  timestamp: string;
}

// WebSocket Types
export interface WebSocketMessage {
  type: 'price_update' | 'portfolio_update' | 'trade_executed' | 'analysis_result' | 'risk_alert' | 'error';
  data: any;
  timestamp: string;
}

export interface PriceUpdate {
  symbol: string;
  price: number;
  timestamp: string;
}

export interface PortfolioUpdate {
  portfolio_id: string;
  data: Portfolio;
  timestamp: string;
}

// Chart Data Types
export interface ChartDataPoint {
  timestamp: string;
  value: number;
  volume?: number;
  open?: number;
  high?: number;
  low?: number;
  close?: number;
}

export interface PnLDataPoint {
  timestamp: string;
  total_pnl: number;
  realized_pnl: number;
  unrealized_pnl: number;
  day_pnl: number;
  total_value: number;
}

// UI State Types
export interface LoadingState {
  isLoading: boolean;
  error: string | null;
}

export interface PaginationState {
  page: number;
  limit: number;
  total: number;
  hasMore: boolean;
}

export interface FilterState {
  symbols: string[];
  timeframe: '1h' | '1d' | '1w' | '1m' | '3m' | '1y';
  sort_by: string;
  sort_order: 'asc' | 'desc';
}

// Rate Limiting Types
export interface RateLimitStatus {
  identifier: string;
  rule: string;
  tier: string;
  limit: number;
  remaining: number;
  used: number;
  reset_time: string;
  reset_time_iso: string;
  burst_limit: number;
  window_seconds: number;
}

// System Health Types
export interface SystemHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  components: {
    [component: string]: string;
  };
  metrics: {
    cpu_usage: number;
    memory_usage: number;
    active_connections: number;
    request_rate_per_second: number;
    error_rate_percent: number;
    avg_response_time_ms: number;
  };
  active_connections: number;
}

// Error Types
export interface APIError {
  error: string;
  message: string;
  status_code: number;
  timestamp: string;
  request_id?: string;
}

// Configuration Types
export interface AppConfig {
  apiUrl: string;
  wsUrl: string;
  environment: 'development' | 'production';
  features: {
    realTimeUpdates: boolean;
    virtualTrading: boolean;
    agentAnalysis: boolean;
    portfolioManagement: boolean;
  };
}

// Component Props Types
export interface ComponentProps {
  className?: string;
  children?: React.ReactNode;
}

export interface ButtonProps extends ComponentProps {
  variant?: 'primary' | 'secondary' | 'success' | 'danger' | 'warning' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  loading?: boolean;
  onClick?: () => void;
  type?: 'button' | 'submit' | 'reset';
}

export interface InputProps extends ComponentProps {
  type?: string;
  placeholder?: string;
  value?: string;
  onChange?: (value: string) => void;
  error?: string;
  disabled?: boolean;
  required?: boolean;
}

export interface ModalProps extends ComponentProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  size?: 'sm' | 'md' | 'lg' | 'xl';
}

// Store Types
export interface AuthState {
  user: User | null;
  tokens: AuthTokens | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  login: (credentials: LoginRequest) => Promise<void>;
  logout: () => void;
  refreshToken: () => Promise<void>;
  clearError: () => void;
}

export interface PortfolioState {
  portfolios: Portfolio[];
  selectedPortfolio: Portfolio | null;
  positions: Position[];
  isLoading: boolean;
  error: string | null;
  fetchPortfolios: () => Promise<void>;
  selectPortfolio: (portfolioId: string) => Promise<void>;
  createPortfolio: (data: any) => Promise<void>;
  updatePortfolio: (portfolioId: string, data: any) => Promise<void>;
  deletePortfolio: (portfolioId: string) => Promise<void>;
  clearError: () => void;
}

export interface TradingState {
  trades: TradeResponse[];
  pendingTrades: TradeRequest[];
  isLoading: boolean;
  error: string | null;
  executeTrade: (trade: TradeRequest) => Promise<TradeResponse>;
  fetchTradeHistory: (portfolioId: string) => Promise<void>;
  cancelTrade: (tradeId: string) => Promise<void>;
  clearError: () => void;
}

export interface MarketDataState {
  quotes: { [symbol: string]: MarketQuote };
  subscriptions: Set<string>;
  isLoading: boolean;
  error: string | null;
  subscribe: (symbols: string[]) => void;
  unsubscribe: (symbols: string[]) => void;
  getQuote: (symbol: string) => MarketQuote | null;
  clearError: () => void;
}

export interface AgentState {
  analyses: AgentAnalysisResponse[];
  currentAnalysis: AgentAnalysisResponse | null;
  isLoading: boolean;
  error: string | null;
  requestAnalysis: (request: AgentAnalysisRequest) => Promise<AgentAnalysisResponse>;
  clearCurrentAnalysis: () => void;
  clearError: () => void;
}

export interface WebSocketState {
  isConnected: boolean;
  connectionStatus: 'disconnected' | 'connecting' | 'connected' | 'error';
  lastMessage: WebSocketMessage | null;
  error: string | null;
  connect: () => void;
  disconnect: () => void;
  sendMessage: (message: any) => void;
  subscribe: (type: string, target: string) => void;
  unsubscribe: (type: string, target: string) => void;
}
