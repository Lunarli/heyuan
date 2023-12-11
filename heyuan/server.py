from http.server import BaseHTTPRequestHandler, HTTPServer

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        message = "This is the GET response"
        self.wfile.write(bytes(message, "utf8"))

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        print(post_data)

        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        message = "This is the POST response" + '\n' + 'The Input is ' + str(post_data)
        self.wfile.write(bytes(message, "utf8"))

def run():
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, RequestHandler)
    print('Starting server...')
    httpd.serve_forever()

if __name__ == '__main__':
    run()