import React, { useContext } from 'react';
import { AuthContext } from '../auth/AuthContext';

const RoleBasedAccess = ({ 
  children, 
  allowedRoles = [], 
  requiredPermissions = [],
  fallback = null 
}) => {
  const { user } = useContext(AuthContext);

  if (!user) {
    return fallback || <div>Please log in to access this feature.</div>;
  }

  // Check role-based access
  if (allowedRoles.length > 0 && !allowedRoles.includes(user.role)) {
    return fallback || <div>You don't have permission to access this feature.</div>;
  }

  // Check permission-based access
  if (requiredPermissions.length > 0) {
    const userPermissions = getUserPermissions(user.role);
    const hasPermission = requiredPermissions.some(permission => 
      userPermissions.includes(permission)
    );
    
    if (!hasPermission) {
      return fallback || <div>You don't have the required permissions for this feature.</div>;
    }
  }

  return children;
};

// Define role-based permissions
const ROLE_PERMISSIONS = {
  admin: [
    'view_all', 'edit_all', 'delete_all', 'manage_users', 'manage_roles',
    'view_reports', 'add_reports', 'edit_reports', 'delete_reports',
    'upload_files', 'send_alerts', 'manage_emergency', 'view_analytics',
    'manage_communities', 'manage_water_quality', 'manage_health_reports'
  ],
  officer: [
    'view_reports', 'add_reports', 'edit_reports', 'upload_files', 
    'send_alerts', 'manage_emergency', 'view_analytics',
    'manage_communities', 'manage_water_quality', 'manage_health_reports'
  ],
  worker: [
    'view_reports', 'add_reports', 'upload_files', 'view_communities',
    'manage_water_quality', 'manage_health_reports'
  ],
  community: [
    'view_reports', 'add_reports', 'view_communities'
  ]
};

const getUserPermissions = (role) => {
  return ROLE_PERMISSIONS[role] || [];
};

// Hook for checking permissions
export const usePermissions = () => {
  const { user } = useContext(AuthContext);
  
  const hasPermission = (permission) => {
    if (!user) return false;
    const userPermissions = getUserPermissions(user.role);
    return userPermissions.includes(permission);
  };

  const hasAnyPermission = (permissions) => {
    if (!user) return false;
    const userPermissions = getUserPermissions(user.role);
    return permissions.some(permission => userPermissions.includes(permission));
  };

  const hasAllPermissions = (permissions) => {
    if (!user) return false;
    const userPermissions = getUserPermissions(user.role);
    return permissions.every(permission => userPermissions.includes(permission));
  };

  const canAccessFeature = (feature) => {
    const featurePermissions = {
      'dashboard': ['view_reports'],
      'water_quality': ['manage_water_quality'],
      'health_reports': ['manage_health_reports'],
      'gis_mapping': ['view_analytics'],
      'emergency_alerts': ['send_alerts', 'manage_emergency'],
      'file_upload': ['upload_files'],
      'user_management': ['manage_users'],
      'analytics': ['view_analytics'],
      'community_management': ['manage_communities']
    };

    const requiredPermissions = featurePermissions[feature] || [];
    return hasAnyPermission(requiredPermissions);
  };

  return {
    hasPermission,
    hasAnyPermission,
    hasAllPermissions,
    canAccessFeature,
    userRole: user?.role,
    userPermissions: getUserPermissions(user?.role)
  };
};

export default RoleBasedAccess;
