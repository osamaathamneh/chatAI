// API Configuration for LLMonDBPGAI Frontend
// This file centralizes the backend API URL configuration

/**
 * API Configuration Object
 * 
 * Change BASE_URL to match your backend server location:
 * - Development: 'http://localhost:8000'
 * - Production: 'https://your-production-backend.com'
 */
window.API_CONFIG = {
    BASE_URL: 'http://localhost:8000',
    
    // Alternative configurations (uncomment as needed):
    
    // For production deployment:
    // BASE_URL: 'https://your-production-backend.com'
    
    // For different local port:
    // BASE_URL: 'http://localhost:5000'
    
    // For network access:
    // BASE_URL: 'http://192.168.1.100:8000'
};

// Log configuration on load (helpful for debugging)
console.log('API Configuration loaded:', window.API_CONFIG.BASE_URL);
