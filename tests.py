import datetime
import itertools
from collections import Counter

import bollinger as B

stock = B.bollinger()

begins = Counter()
ends = Counter()
ends
for transaction in stock.transactions:
    # begins[transaction.signal_begin.index] += 1
    ends[transaction.signal_end.index] += 1
    print(
        f"begin = {transaction.signal_begin.date}\nend = {transaction.signal_end.date}")
