
from datetime import datetime
def format_msg(msg):
    return f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
