# Recipe: Streaming server

Run the server with more worker threads by editing `serving/http.py` to pass
`ThreadingHTTPServer` instead of `HTTPServer` for concurrent requests.
