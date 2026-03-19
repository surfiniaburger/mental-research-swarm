import http.server
import socketserver
import json
import os
import sys

PORT = 8080
LOG_FILE = "telemetry.log"
DASHBOARD_FILE = "dashboard/index.html"

class TelemetryHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            if os.path.exists(DASHBOARD_FILE):
                with open(DASHBOARD_FILE, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.wfile.write(b"Dashboard file not found. Run from the project root.")
        
        elif self.path == '/api/data':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            data = []
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, 'r') as f:
                    for line in f:
                        try:
                            data.append(json.loads(line))
                        except:
                            continue
            self.wfile.write(json.dumps(data).encode())
        else:
            super().do_GET()

def main():
    # Change to root dir if needed
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    with socketserver.TCPServer(("", PORT), TelemetryHandler) as httpd:
        print(f"🚀 Telemetry Dashboard Live at http://localhost:{PORT}")
        print("Monitoring telemetry.log...")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
            httpd.server_close()

if __name__ == "__main__":
    main()
