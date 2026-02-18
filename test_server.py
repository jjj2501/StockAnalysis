#!/usr/bin/env python3
"""
Simple test server for StockAnalysis AI interface
This server provides mock API endpoints for testing the frontend
without requiring complex dependencies.
"""

import http.server
import socketserver
import json
import sqlite3
import os
import re
import hashlib
import secrets
from urllib.parse import urlparse, parse_qs
from datetime import datetime, timedelta

# Database setup
DB_FILE = "stockanalysis_test.db"

def init_database():
    """Initialize SQLite database for testing"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create watchlist table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            name TEXT,
            price REAL,
            change REAL,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            UNIQUE(user_id, symbol)
        )
    ''')
    
    conn.commit()
    conn.close()

def hash_password(password):
    """Simple password hashing for testing"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, password_hash):
    """Verify password against hash"""
    return hash_password(password) == password_hash

def generate_token(username):
    """Generate a simple token for authentication"""
    return hashlib.sha256(f"{username}{secrets.token_hex(16)}".encode()).hexdigest()

# Mock stock data
MOCK_STOCKS = {
    "AAPL": {
        "name": "Apple Inc.",
        "price": 182.63,
        "change": 1.24,
        "change_percent": 0.68,
        "volume": 58392000,
        "market_cap": "2.87T",
        "pe_ratio": 28.5,
        "dividend_yield": 0.51,
        "sector": "Technology",
        "description": "Apple designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide."
    },
    "MSFT": {
        "name": "Microsoft Corporation",
        "price": 415.86,
        "change": 3.21,
        "change_percent": 0.78,
        "volume": 25431000,
        "market_cap": "3.09T",
        "pe_ratio": 35.2,
        "dividend_yield": 0.71,
        "sector": "Technology",
        "description": "Microsoft develops, licenses, and supports software, services, devices, and solutions worldwide."
    },
    "GOOGL": {
        "name": "Alphabet Inc.",
        "price": 151.23,
        "change": -0.87,
        "change_percent": -0.57,
        "volume": 31245000,
        "market_cap": "1.91T",
        "pe_ratio": 26.8,
        "dividend_yield": 0.0,
        "sector": "Communication Services",
        "description": "Alphabet provides various products and platforms in the United States, Europe, the Middle East, Africa, the Asia-Pacific, Canada, and Latin America."
    },
    "AMZN": {
        "name": "Amazon.com Inc.",
        "price": 178.21,
        "change": 2.34,
        "change_percent": 1.33,
        "volume": 42318000,
        "market_cap": "1.83T",
        "pe_ratio": 60.5,
        "dividend_yield": 0.0,
        "sector": "Consumer Cyclical",
        "description": "Amazon engages in the retail sale of consumer products and subscriptions through online and physical stores in North America and internationally."
    },
    "TSLA": {
        "name": "Tesla Inc.",
        "price": 177.90,
        "change": -3.21,
        "change_percent": -1.77,
        "volume": 102345000,
        "market_cap": "567.2B",
        "pe_ratio": 42.3,
        "dividend_yield": 0.0,
        "sector": "Consumer Cyclical",
        "description": "Tesla designs, develops, manufactures, leases, and sells electric vehicles, and energy generation and storage systems in the United States, China, and internationally."
    },
    "NVDA": {
        "name": "NVIDIA Corporation",
        "price": 903.56,
        "change": 15.32,
        "change_percent": 1.72,
        "volume": 48321000,
        "market_cap": "2.26T",
        "pe_ratio": 76.4,
        "dividend_yield": 0.02,
        "sector": "Technology",
        "description": "NVIDIA provides graphics, compute, and networking solutions in the United States, Taiwan, China, and internationally."
    }
}

class TestServerHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler for test server"""
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        # Serve static files from frontend directory
        if path == "/" or path.endswith(".html") or path.endswith(".css") or path.endswith(".js"):
            if path == "/":
                path = "/index.html"
            
            # Map paths to frontend directory
            file_path = f"frontend{path}"
            if os.path.exists(file_path):
                self.serve_static_file(file_path)
            else:
                self.send_error(404, f"File not found: {path}")
            return
        
        # API endpoints
        if path == "/api/health":
            self.send_json_response({"status": "ok", "message": "Server is running"})
        
        elif path == "/api/stocks":
            self.send_json_response({"stocks": list(MOCK_STOCKS.keys())})
        
        elif path.startswith("/api/stock/"):
            symbol = path.split("/")[-1].upper()
            if symbol in MOCK_STOCKS:
                self.send_json_response({"stock": MOCK_STOCKS[symbol]})
            else:
                self.send_error(404, f"Stock not found: {symbol}")
        
        else:
            self.send_error(404, f"Endpoint not found: {path}")
    
    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        # Read request body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')
        
        try:
            data = json.loads(body) if body else {}
        except:
            data = {}
        
        # API endpoints
        if path == "/api/register":
            self.handle_register(data)
        
        elif path == "/api/login":
            self.handle_login(data)
        
        elif path == "/api/logout":
            self.send_json_response({"message": "Logged out successfully"})
        
        elif path == "/api/analyze":
            self.handle_analyze(data)
        
        elif path == "/api/watchlist":
            self.handle_watchlist(data)
        
        elif path == "/api/settings":
            self.send_json_response({"message": "Settings updated successfully"})
        
        else:
            self.send_error(404, f"Endpoint not found: {path}")
    
    def serve_static_file(self, file_path):
        """Serve static file with appropriate content type"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Determine content type
            if file_path.endswith('.html'):
                content_type = 'text/html'
            elif file_path.endswith('.css'):
                content_type = 'text/css'
            elif file_path.endswith('.js'):
                content_type = 'application/javascript'
            else:
                content_type = 'text/plain'
            
            self.send_response(200)
            self.send_header('Content-type', content_type)
            self.send_header('Content-length', str(len(content)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(content)
        
        except Exception as e:
            self.send_error(500, f"Error reading file: {str(e)}")
    
    def send_json_response(self, data, status=200):
        """Send JSON response"""
        response = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Content-length', str(len(response)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(response)
    
    def send_error_response(self, code, message):
        """Send error response"""
        self.send_json_response({"error": message}, code)
    
    def send_error(self, code, message, explain=None):
        """Override send_error to send JSON responses"""
        self.send_error_response(code, message)
    
    def handle_register(self, data):
        """Handle user registration"""
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        full_name = data.get('full_name', '').strip()
        
        # Validation
        if not username or not email or not password:
            self.send_error_response(400, "Username, email, and password are required")
            return
        
        if len(password) < 6:
            self.send_error_response(400, "Password must be at least 6 characters")
            return
        
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            
            # Check if user exists
            cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
            if cursor.fetchone():
                self.send_error_response(400, "Username or email already exists")
                return
            
            # Create user
            password_hash = hash_password(password)
            cursor.execute(
                "INSERT INTO users (username, email, password_hash, full_name) VALUES (?, ?, ?, ?)",
                (username, email, password_hash, full_name)
            )
            user_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            
            # Generate token
            token = generate_token(username)
            
            self.send_json_response({
                "message": "Registration successful",
                "user": {
                    "id": user_id,
                    "username": username,
                    "email": email,
                    "full_name": full_name
                },
                "token": token
            })
        
        except Exception as e:
            self.send_error_response(500, f"Registration failed: {str(e)}")
    
    def handle_login(self, data):
        """Handle user login"""
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if not username or not password:
            self.send_error_response(400, "Username and password are required")
            return
        
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            
            # Find user
            cursor.execute(
                "SELECT id, username, email, password_hash, full_name FROM users WHERE username = ? OR email = ?",
                (username, username)
            )
            user = cursor.fetchone()
            
            if not user:
                self.send_error_response(401, "Invalid username or password")
                return
            
            user_id, db_username, email, password_hash, full_name = user
            
            # Verify password
            if not verify_password(password, password_hash):
                self.send_error_response(401, "Invalid username or password")
                return
            
            # Generate token
            token = generate_token(username)
            
            conn.close()
            
            self.send_json_response({
                "message": "Login successful",
                "user": {
                    "id": user_id,
                    "username": db_username,
                    "email": email,
                    "full_name": full_name
                },
                "token": token
            })
        
        except Exception as e:
            self.send_error_response(500, f"Login failed: {str(e)}")
    
    def handle_analyze(self, data):
        """Handle stock analysis request"""
        symbol = data.get('symbol', '').upper().strip()
        
        if not symbol:
            self.send_error_response(400, "Stock symbol is required")
            return
        
        if symbol not in MOCK_STOCKS:
            self.send_error_response(404, f"Stock not found: {symbol}")
            return
        
        stock_data = MOCK_STOCKS[symbol]
        
        # Generate mock analysis
        analysis = {
            "symbol": symbol,
            "name": stock_data["name"],
            "current_price": stock_data["price"],
            "analysis": {
                "technical": {
                    "rating": "BUY" if stock_data["change"] > 0 else "HOLD",
                    "rsi": 65 if stock_data["change"] > 0 else 45,
                    "macd": "Bullish" if stock_data["change"] > 0 else "Neutral",
                    "support": stock_data["price"] * 0.95,
                    "resistance": stock_data["price"] * 1.05
                },
                "fundamental": {
                    "rating": "STRONG_BUY" if stock_data["pe_ratio"] < 30 else "HOLD",
                    "pe_ratio": stock_data["pe_ratio"],
                    "market_cap": stock_data["market_cap"],
                    "dividend_yield": stock_data["dividend_yield"],
                    "sector": stock_data["sector"]
                },
                "sentiment": {
                    "rating": "POSITIVE" if stock_data["change"] > 0 else "NEUTRAL",
                    "news_sentiment": 0.7 if stock_data["change"] > 0 else 0.5,
                    "social_mentions": 1250,
                    "institutional_ownership": "78%"
                }
            },
            "recommendation": "BUY" if stock_data["change"] > 0 and stock_data["pe_ratio"] < 30 else "HOLD",
            "confidence": 0.85 if stock_data["change"] > 0 else 0.65,
            "timestamp": datetime.now().isoformat()
        }
        
        self.send_json_response(analysis)
    
    def handle_watchlist(self, data):
        """Handle watchlist operations"""
        action = data.get('action', '')
        symbol = data.get('symbol', '').upper().strip()
        user_id = data.get('user_id', 1)  # Default to user 1 for testing
        
        if action in ['add', 'remove']:
            if not symbol or symbol not in MOCK_STOCKS:
                self.send_error_response(400, "Invalid stock symbol")
                return
        elif action == 'list':
            # Skip symbol validation for list action
            pass
        else:
            self.send_error_response(400, "Invalid action")
            return
        
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            
            if action == 'add':
                stock_data = MOCK_STOCKS[symbol]
                # Add to watchlist
                cursor.execute(
                    "INSERT OR IGNORE INTO watchlist (user_id, symbol, name, price, change) VALUES (?, ?, ?, ?, ?)",
                    (user_id, symbol, stock_data["name"], stock_data["price"], stock_data["change"])
                )
                message = f"Added {symbol} to watchlist"
            
            elif action == 'remove':
                # Remove from watchlist
                cursor.execute(
                    "DELETE FROM watchlist WHERE user_id = ? AND symbol = ?",
                    (user_id, symbol)
                )
                message = f"Removed {symbol} from watchlist"
            
            elif action == 'list':
                # List watchlist
                cursor.execute(
                    "SELECT symbol, name, price, change, added_at FROM watchlist WHERE user_id = ? ORDER BY added_at DESC",
                    (user_id,)
                )
                items = []
                for row in cursor.fetchall():
                    symbol, name, price, change, added_at = row
                    items.append({
                        "symbol": symbol,
                        "name": name,
                        "price": price,
                        "change": change,
                        "added_at": added_at
                    })
                
                self.send_json_response({"watchlist": items})
                conn.close()
                return
            
            else:
                self.send_error(400, "Invalid action")
                return
            
            conn.commit()
            conn.close()
            
            self.send_json_response({"message": message})
        
        except Exception as e:
            self.send_error_response(500, f"Watchlist operation failed: {str(e)}")

def main():
    """Main function to start the test server"""
    # Initialize database
    init_database()
    
    # Create default test user
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Check if test user exists
    cursor.execute("SELECT id FROM users WHERE username = 'testuser'")
    if not cursor.fetchone():
        password_hash = hash_password("password123")
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, full_name) VALUES (?, ?, ?, ?)",
            ("testuser", "test@example.com", password_hash, "Test User")
        )
        conn.commit()
    
    conn.close()
    
    # Start server
    PORT = 8080
    Handler = TestServerHandler
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Test server running at http://localhost:{PORT}")
        print("Available endpoints:")
        print("  GET  /api/health - Server health check")
        print("  GET  /api/stocks - List available stocks")
        print("  GET  /api/stock/{symbol} - Get stock details")
        print("  POST /api/register - Register new user")
        print("  POST /api/login - User login")
        print("  POST /api/logout - User logout")
        print("  POST /api/analyze - Analyze stock")
        print("  POST /api/watchlist - Manage watchlist")
        print("  POST /api/settings - Update user settings")
        print("\nTest credentials:")
        print("  Username: testuser")
        print("  Password: password123")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    main()