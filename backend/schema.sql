-- PostgreSQL Schema for Vitanza Health Surveillance System
-- This schema provides an alternative to MongoDB for production deployment

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    display_name VARCHAR(255),
    role VARCHAR(50) DEFAULT 'community',
    email_verified BOOLEAN DEFAULT FALSE,
    verified_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Email verifications table
CREATE TABLE IF NOT EXISTS email_verifications (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Communities table
CREATE TABLE IF NOT EXISTS communities (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    district VARCHAR(255),
    state VARCHAR(255),
    population INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Health reports table
CREATE TABLE IF NOT EXISTS health_reports (
    id SERIAL PRIMARY KEY,
    patient_id VARCHAR(255) NOT NULL,
    symptoms TEXT[],
    location VARCHAR(255),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Water quality reports table
CREATE TABLE IF NOT EXISTS water_quality_reports (
    id SERIAL PRIMARY KEY,
    sensor_id VARCHAR(255) NOT NULL,
    ph DECIMAL(5,2),
    turbidity DECIMAL(8,2),
    tds DECIMAL(8,2),
    temp DECIMAL(5,2),
    location VARCHAR(255),
    risk VARCHAR(20),
    risk_proba DECIMAL(5,4),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Alerts table
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    payload JSONB,
    user_id INTEGER REFERENCES users(id),
    sent_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Notifications table
CREATE TABLE IF NOT EXISTS notifications (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    user_id INTEGER REFERENCES users(id),
    severity VARCHAR(20) DEFAULT 'info',
    read BOOLEAN DEFAULT FALSE,
    sent_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_email_verifications_token ON email_verifications(token);
CREATE INDEX IF NOT EXISTS idx_health_reports_timestamp ON health_reports(timestamp);
CREATE INDEX IF NOT EXISTS idx_water_reports_timestamp ON water_quality_reports(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at);
CREATE INDEX IF NOT EXISTS idx_notifications_user_id ON notifications(user_id);

-- Insert default admin user (password: admin123)
INSERT INTO users (email, password_hash, display_name, role, email_verified) 
VALUES ('admin@vitanza.buzz', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj4J/HSy7K2O', 'Admin User', 'admin', TRUE)
ON CONFLICT (email) DO NOTHING;

-- Insert sample communities
INSERT INTO communities (name, district, state, population) VALUES
('Village A', 'District 1', 'State 1', 1500),
('Village B', 'District 2', 'State 1', 2000),
('Village C', 'District 1', 'State 2', 1200)
ON CONFLICT DO NOTHING;
