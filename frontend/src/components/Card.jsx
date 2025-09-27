import React from 'react';

const Card = ({ children, style = {}, ...props }) => {
  const defaultStyle = {
    backgroundColor: 'white',
    borderRadius: '0.5rem',
    boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
    border: '1px solid #e5e7eb',
    ...style
  };

  return (
    <div 
      style={defaultStyle}
      {...props}
    >
      {children}
    </div>
  );
};

export default Card;
