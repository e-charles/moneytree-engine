# import threading


# _FALLBACK_USER_AGENT = (
#     "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
#     "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
# )


# def new_session(): 
#     """ Create def Session for the active backend."""
#     s = 
#     s.headers.update({
#         "User-Agent": _FALLBACK_USER_AGENT,
#         "Accept": "text/html,application/xhtml+xml,application/xml;q-0.9,*/*;q=0.8",
#         "Accept-Language": "en-US,en;q=0.5"
#     })
#     return s


# from curl_cffi import Session, WebSocketRetryStrategy
# from datetime import timedelta

# import pprint

# url = 'https://etfdb.com'

# # strategy = WebSocketRetryStrategy(count=3, delay=0.2, jitter=0.1, backoff="exponential")
# with Session(cache=timedelta(minutes=5), retry=3) as s:
#     r = s.get(url)
#     pprint.pprint(r.cookies)

import yfinance as yf
