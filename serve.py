import http.server, socketserver, os
os.chdir(os.path.join(os.path.dirname(__file__), "docs"))
class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()
    def log_message(self, *a): pass
class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True; allow_reuse_address = True
S(("0.0.0.0", 7860), H).serve_forever()
