import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  LayoutDashboard,
  Briefcase,
  TrendingUp,
  Bot,
  BarChart3,
  Shield,
  Settings,
  HelpCircle,
} from 'lucide-react';
import { clsx } from 'clsx';

interface SidebarItem {
  name: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
  badge?: string;
}

const navigationItems: SidebarItem[] = [
  { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
  { name: 'Portfolios', href: '/portfolios', icon: Briefcase },
  { name: 'Trading', href: '/trading', icon: TrendingUp },
  { name: 'AI Analysis', href: '/analysis', icon: Bot, badge: 'AI' },
  { name: 'Market Data', href: '/market', icon: BarChart3 },
  { name: 'Risk Management', href: '/risk', icon: Shield },
];

const secondaryItems: SidebarItem[] = [
  { name: 'Settings', href: '/settings', icon: Settings },
  { name: 'Help & Support', href: '/help', icon: HelpCircle },
];

const Sidebar: React.FC = () => {
  const location = useLocation();

  const isActive = (href: string) => {
    return location.pathname === href || location.pathname.startsWith(href + '/');
  };

  return (
    <div className="w-64 bg-white border-r border-gray-200 flex flex-col">
      <div className="flex-1 flex flex-col pt-5 pb-4 overflow-y-auto">
        <nav className="mt-5 flex-1 px-2 space-y-1">
          {navigationItems.map((item) => {
            const active = isActive(item.href);
            
            return (
              <Link
                key={item.name}
                to={item.href}
                className={clsx(
                  'group flex items-center px-3 py-2 text-sm font-medium rounded-md transition-all duration-200',
                  active
                    ? 'bg-primary-50 text-primary-700 border-r-2 border-primary-600'
                    : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                )}
              >
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="flex items-center w-full"
                >
                  <item.icon
                    className={clsx(
                      'flex-shrink-0 w-5 h-5 mr-3',
                      active ? 'text-primary-600' : 'text-gray-400 group-hover:text-gray-500'
                    )}
                  />
                  <span className="flex-1">{item.name}</span>
                  {item.badge && (
                    <span className="ml-2 px-2 py-0.5 text-xs font-medium bg-primary-100 text-primary-700 rounded-full">
                      {item.badge}
                    </span>
                  )}
                </motion.div>
              </Link>
            );
          })}
        </nav>

        {/* Secondary Navigation */}
        <nav className="mt-8 px-2 space-y-1 border-t border-gray-200 pt-6">
          {secondaryItems.map((item) => {
            const active = isActive(item.href);
            
            return (
              <Link
                key={item.name}
                to={item.href}
                className={clsx(
                  'group flex items-center px-3 py-2 text-sm font-medium rounded-md transition-all duration-200',
                  active
                    ? 'bg-primary-50 text-primary-700'
                    : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                )}
              >
                <item.icon
                  className={clsx(
                    'flex-shrink-0 w-5 h-5 mr-3',
                    active ? 'text-primary-600' : 'text-gray-400 group-hover:text-gray-500'
                  )}
                />
                {item.name}
              </Link>
            );
          })}
        </nav>
      </div>

      {/* Footer */}
      <div className="flex-shrink-0 px-4 py-3 border-t border-gray-200">
        <div className="text-xs text-gray-500 text-center">
          <p>MIP v1.0.0</p>
          <p>Multi-Agent Intelligence</p>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
