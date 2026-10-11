import unittest
import threading
import json
import urllib.request
from strataforge.config import Config
from strataforge.serving.http import create_app
from http.server import HTTPServer


class TestHTTP(unittest.TestCase):
    def test_endpoints(self):
        cfg = Config()
        handler = create_app(cfg)
        httpd = HTTPServer(("127.0.0.1", 0), handler)
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=10) as r:
                self.assertEqual(r.status, 200)
            body = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
            req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                         data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read().decode())
                self.assertIn("choices", data)
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
