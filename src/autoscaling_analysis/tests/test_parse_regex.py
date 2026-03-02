from autoscaling_analysis.ingest.schemas import LOG_RE, REQ_RE

def test_log_re_matches_sample():
    line = '127.0.0.1 - - [01/Jul/1995:00:00:01 -0400] "GET /index.html HTTP/1.0" 200 123'
    m = LOG_RE.match(line)
    assert m is not None
    assert m.group("host") == "127.0.0.1"
    assert m.group("status") == "200"

def test_req_re_matches_sample():
    req = "GET /index.html HTTP/1.0"
    m = REQ_RE.match(req)
    assert m is not None
    assert m.group("method") == "GET"