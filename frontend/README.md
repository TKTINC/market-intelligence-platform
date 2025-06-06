# Enhanced React Dashboard - MIP

The Enhanced React Dashboard is a modern, real-time trading interface for the Multi-Agent Market Intelligence Platform (MIP). Built with React 18, TypeScript, and Tailwind CSS, it provides a comprehensive user experience for AI-powered market analysis and virtual trading.

## Features

### 🤖 AI Agent Integration
- **Agent Selection Interface** - Choose from FinBERT, TFT, GPT-4, and Llama agents
- **Cost-Aware Analysis** - Track analysis costs with user tier limits
- **Real-time Results** - Live streaming of analysis results
- **Interactive Explanations** - Human-readable market insights

### 💰 Virtual Trading Platform
- **Real-time Trading** - Execute market, limit, and stop orders
- **Portfolio Management** - Create and manage multiple portfolios
- **Live P&L Tracking** - Real-time profit/loss calculations
- **Risk Management** - Position limits and risk alerts

### 📊 Real-time Market Data
- **Live Price Feeds** - WebSocket-based real-time quotes
- **Interactive Charts** - Advanced price and P&L visualizations
- **Market Overview** - Comprehensive market dashboard
- **Performance Analytics** - Portfolio performance metrics

### 🔒 Enterprise Security
- **JWT Authentication** - Secure token-based login
- **Role-based Access** - User tier permissions (Free/Basic/Premium/Enterprise)
- **Rate Limiting** - Smart API usage controls
- **Session Management** - Secure session handling

## Tech Stack

### Core Technologies
- **React 18** - Modern React with hooks and concurrent features
- **TypeScript** - Type-safe development
- **Vite** - Fast build tool and dev server
- **Tailwind CSS** - Utility-first CSS framework

### State Management
- **Zustand** - Lightweight state management
- **React Hook Form** - Form handling with validation
- **Zod** - Runtime type validation

### UI/UX
- **Framer Motion** - Smooth animations and transitions
- **Lucide React** - Beautiful icon library
- **React Hot Toast** - Toast notifications
- **Recharts** - Responsive chart library

### Networking
- **Axios** - HTTP client with interceptors
- **WebSocket** - Real-time data streaming
- **React Query** - Server state management

## Project Structure

```
src/
├── components/          # Reusable UI components
│   ├── ui/             # Basic UI components (Button, Input, Modal)
│   ├── layout/         # Layout components (Header, Sidebar)
│   ├── dashboard/      # Dashboard-specific components
│   ├── trading/        # Trading interface components
│   ├── portfolio/      # Portfolio management components
│   └── analysis/       # AI analysis components
├── pages/              # Page components
├── store/              # Zustand stores
├── services/           # API and WebSocket services
├── types/              # TypeScript type definitions
├── utils/              # Utility functions
└── hooks/              # Custom React hooks
```

## Getting Started

### Prerequisites
- Node.js 18+ 
- npm or yarn
- Enhanced FastAPI Gateway running on port 8000

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mip-react-dashboard
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env.development
   ```
   
   Update the environment variables:
   ```env
   VITE_API_URL=http://localhost:8000
   VITE_WS_URL=ws://localhost:8000/ws
   VITE_ENV=development
   ```

4. **Start the development server**
   ```bash
   npm run dev
   ```

5. **Open your browser**
   Navigate to `http://localhost:3000`

### Demo Credentials
- **Email:** demo@mip.com
- **Password:** demo123

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint
- `npm run type-check` - Run TypeScript compiler
- `npm run format` - Format code with Prettier

## Key Components

### Authentication
- JWT-based authentication with automatic token refresh
- Secure session management with localStorage persistence
- Role-based route protection

### Trading Interface
- Real-time order execution with market simulation
- Support for market, limit, and stop orders
- Live P&L updates via WebSocket
- Risk validation before trade execution

### Portfolio Management
- Create and manage multiple virtual portfolios
- Real-time position tracking
- Portfolio performance analytics
- Risk metrics and alerts

### AI Analysis
- Interactive agent selection with cost visualization
- Real-time analysis streaming
- Results visualization and export
- Historical analysis tracking

### Real-time Features
- WebSocket connections for live data
- Automatic reconnection and error handling
- Real-time portfolio updates
- Live market data streaming

## API Integration

The dashboard integrates with the Enhanced FastAPI Gateway:

### Authentication Endpoints
- `POST /auth/login` - User login
- `POST /auth/refresh` - Token refresh
- `POST /auth/logout` - User logout

### Portfolio Endpoints
- `GET /portfolios/user/me` - Get user portfolios
- `POST /portfolios/create` - Create portfolio
- `GET /portfolios/{id}` - Get portfolio details

### Trading Endpoints
- `POST /trading/execute` - Execute trade
- `GET /trading/history/{portfolio_id}` - Get trade history

### Analysis Endpoints
- `POST /agents/analyze` - Request AI analysis
- `GET /agents/status` - Get agent status

### WebSocket Endpoints
- `/ws/user/{user_id}` - User-specific updates
- `/ws/portfolio/{portfolio_id}` - Portfolio updates
- `/ws/market/{symbols}` - Market data updates

## Deployment

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t mip-dashboard .
   ```

2. **Run the container**
   ```bash
   docker run -p 3000:80 mip-dashboard
   ```

### Production Build

1. **Build for production**
   ```bash
   npm run build
   ```

2. **Serve with nginx**
   The built files in `dist/` can be served with any static file server.

## Configuration

### Environment Variables
- `VITE_API_URL` - FastAPI Gateway URL
- `VITE_WS_URL` - WebSocket URL
- `VITE_ENV` - Environment (development/production)

### Build Configuration
- Vite configuration in `vite.config.ts`
- Tailwind configuration in `tailwind.config.js`
- TypeScript configuration in `tsconfig.json`

## Performance Optimizations

- **Code Splitting** - Automatic route-based code splitting
- **Tree Shaking** - Remove unused code in production
- **Asset Optimization** - Optimized images and fonts
- **Caching** - Aggressive caching for static assets
- **Bundle Analysis** - Bundle size monitoring

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.
