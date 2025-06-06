import React from 'react';
import { clsx } from 'clsx';
import { InputProps } from '@/types';

const Input: React.FC<InputProps> = ({
  type = 'text',
  placeholder,
  value,
  onChange,
  error,
  disabled = false,
  required = false,
  className,
  ...props
}) => {
  const baseClasses = 'block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 disabled:bg-gray-50 disabled:text-gray-500 disabled:cursor-not-allowed sm:text-sm';
  
  const errorClasses = error 
    ? 'border-danger-300 text-danger-900 placeholder-danger-300 focus:border-danger-500 focus:ring-danger-500'
    : '';

  return (
    <div className="relative">
      <input
        type={type}
        placeholder={placeholder}
        value={value}
        onChange={(e) => onChange?.(e.target.value)}
        disabled={disabled}
        required={required}
        className={clsx(baseClasses, errorClasses, className)}
        {...props}
      />
      {error && (
        <p className="mt-1 text-sm text-danger-600">
          {error}
        </p>
      )}
    </div>
  );
};

export default Input;
