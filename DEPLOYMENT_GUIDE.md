# Vitanza Deployment Guide

## Quick Deployment (1 Hour)

### Prerequisites
- Node.js 18+ and npm
- Python 3.9+ and pip
- PostgreSQL 14+ (or MongoDB)
- SMTP credentials (Gmail recommended)

### Environment Variables

Create `.env` files:

**Backend (.env):**
```env
# Database
MONGODB_URI=mongodb://localhost:27017/vitanza
# OR for PostgreSQL:
# DATABASE_URL=postgresql://user:password@localhost:5432/vitanza

# JWT
JWT_SECRET=your-super-secret-jwt-key-here
JWT_EXP_MIN=240

# Email (Gmail)
EMAIL_PASSWORD=your-gmail-app-password

# CORS
ALLOWED_ORIGINS=http://localhost:3000,https://vitanza.buzz
```

**Frontend (.env):**
```env
VITE_API_BASE_URL=http://localhost:8000
```

### Backend Setup

1. **Install dependencies:**
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

2. **Database setup:**
```bash
# For PostgreSQL:
psql -U postgres -c "CREATE DATABASE vitanza;"
psql -U postgres -d vitanza -f schema.sql

# For MongoDB:
# MongoDB will auto-create collections
```

3. **Start backend:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Frontend Setup

1. **Install dependencies:**
```bash
cd frontend
npm install
```

2. **Start frontend:**
```bash
npm run dev
```

### Production Deployment

#### Option 1: Docker (Recommended)

**Dockerfile (Backend):**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Dockerfile (Frontend):**
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
FROM nginx:alpine
COPY --from=0 /app/dist /usr/share/nginx/html
```

**docker-compose.yml:**
```yaml
version: '3.8'
services:
  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: vitanza
      POSTGRES_USER: vitanza
      POSTGRES_PASSWORD: vitanza123
    volumes:
      - postgres_data:/var/lib/postgresql/data

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://vitanza:vitanza123@postgres:5432/vitanza
    depends_on:
      - postgres

  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend

volumes:
  postgres_data:
```

#### Option 2: Cloud Deployment

**Vercel (Frontend):**
```bash
npm install -g vercel
cd frontend
vercel --prod
```

**Railway/Render (Backend):**
```bash
# Connect GitHub repo
# Set environment variables
# Deploy
```

### Email Configuration

1. **Gmail Setup:**
   - Enable 2-factor authentication
   - Generate App Password
   - Use: `vitanza.buzz@gmail.com`
   - Password: Your app password

2. **SMTP Settings:**
   - Server: smtp.gmail.com
   - Port: 587
   - Security: STARTTLS

### Domain Setup

1. **DNS Configuration:**
   - A record: vitanza.buzz → Server IP
   - CNAME: www.vitanza.buzz → vitanza.buzz

2. **SSL Certificate:**
   - Use Let's Encrypt (free)
   - Or Cloudflare (automatic)

### Monitoring & Maintenance

1. **Health Checks:**
   - Backend: `GET /health-check`
   - Frontend: Static files served

2. **Logs:**
   - Backend: Check uvicorn logs
   - Frontend: Check nginx logs

3. **Backups:**
   - Database: Daily automated backups
   - Files: Git repository

### Performance Optimization

1. **Database:**
   - Add indexes for frequently queried fields
   - Use connection pooling
   - Monitor query performance

2. **Frontend:**
   - Enable gzip compression
   - Use CDN for static assets
   - Implement caching headers

3. **Backend:**
   - Use async/await properly
   - Implement rate limiting
   - Cache frequently accessed data

### Security Checklist

- [ ] Change default passwords
- [ ] Enable HTTPS
- [ ] Set up CORS properly
- [ ] Validate all inputs
- [ ] Use environment variables for secrets
- [ ] Regular security updates
- [ ] Monitor for suspicious activity

### Troubleshooting

**Common Issues:**

1. **Database Connection:**
   - Check connection string
   - Verify database is running
   - Check firewall settings

2. **Email Not Sending:**
   - Verify SMTP credentials
   - Check spam folder
   - Test with different email provider

3. **CORS Errors:**
   - Update ALLOWED_ORIGINS
   - Check frontend URL

4. **Build Failures:**
   - Clear node_modules and reinstall
   - Check Python version compatibility
   - Verify all dependencies

### Support

For deployment issues:
- Check logs first
- Verify environment variables
- Test each component individually
- Use health check endpoints

**Emergency Contacts:**
- Technical: [Your contact]
- Database: [DBA contact]
- Infrastructure: [DevOps contact]
