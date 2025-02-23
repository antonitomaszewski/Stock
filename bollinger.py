import time

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go


def timeit(f):
    """
    simple decorator for measuring time of function execution
    """
    def _timeit(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print(f'Time of execution : {round(end-start, 2)} seconds')
        return result
    return _timeit


class Stock:
    """
    Holds all information about stock prices, and history of all our trades
    """

    def __init__(self, file='BTC.csv', k=2, n=20,
                 time_frame=1000,  time_step=1, money=100,
                 commission=.998):
        self.stock_df = Stock.process_file(file, k, n, time_frame, time_step)
        self.transactions = []
        self.k = k
        self.n = n
        self.time_frame = time_frame
        self.time_step = time_step
        self.money = money
        self.commission = commission

    def process_file(file, k, n, time_frame, time_step):
        stock_df = pd.read_csv('BTC.csv', parse_dates=[
                               'open_time']).loc[::time_step]
        # stock_df = stock_df[-time_frame - n:]
        stock_df['ma1'] = Stock.MA(stock_df['open'], span=1)
        stock_df['avg'] = Stock.MA(stock_df['open'], span=n)
        stock_df[f'std{k}'] = Stock.std(stock_df['open'], span=n)
        stock_df['upper'] = stock_df['avg'] + k * stock_df[f'std{k}']
        stock_df['lower'] = stock_df['avg'] - k * stock_df[f'std{k}']
        stock_df = stock_df[-time_frame:]
        return stock_df.reset_index(drop=True)

    def MA(series, span):
        return series.rolling(span).mean()

    def std(series, span):
        return series.rolling(span).std()

    def get_row(self, index):
        return self.stock_df.loc[[index]]

    def __iter__(self):
        self.index = self.stock_df.index.start
        return self

    def __next__(self):
        if self.index < self.stock_df.index.stop:
            result = Signal(self.get_row(self.index))
            self.index += self.stock_df.index.step
            return result
        else:
            raise StopIteration

    def __repr__(self):
        return f'''money: {round(self.money, 3)}\ntransactions: {len(self.transactions)}'''

    def make_transaction(self, signal_begin, signal_end):
        history = self.get_history(signal_begin)
        transaction = Transaction(history, signal_begin, signal_end,
                                  self.money, self.commission)
        self.transactions.append(transaction)
        self.money *= transaction.rate

    def get_history(self, signal_begin):
        return self.stock_df.loc[signal_begin.index - self.n + 1:
                                 signal_begin.index]

    def get_data_for_ML(self):
        y = pd.DataFrame(
            (transaction.profitability for transaction in self.transactions)
        )
        X = pd.DataFrame((
            list(transaction.history.open / transaction.history.open.iloc[-1])
            + list(transaction.history.avg / transaction.history.open.iloc[-1])
            + list(transaction.history.lower /
                   transaction.history.open.iloc[-1])
            for transaction in self.transactions
        ))
        return X, y

    @property
    def signal_iterator(self):
        for i in range(self.index + self.stock_df.index.step, self.stock_df.index.stop, self.stock_df.index.step):
            yield Signal(self.get_row(i))


class Signal:
    """
    Represents single row of Stock data
    """
    index = property()
    date = property()

    def __init__(self, row):
        self.row = row

    @index.getter
    def index(self):
        return self.row.index[0]

    @date.getter
    def date(self):
        return self.row.open_time

    def get_price(self, column='open'):
        return float(self.row[column])

    def is_buy_start(self):
        return self.get_price('open') < self.get_price('lower')

    def is_buy_stop(self):
        return self.get_price('open') > self.get_price('avg')

    def __repr__(self):
        return f'{self.row}'


class Transaction:
    """
    Represents transaction buy->sell,
    history will be used for machine learning algorithms to teach them how data
     'looks' before profitable purchase of stock
    """

    def __init__(self, history, signal_begin, signal_end,
                 money, commission):
        self.history = history
        self.signal_begin = signal_begin
        self.signal_end = signal_end
        self.rate = commission * signal_end.get_price() / signal_begin.get_price()
        self.profitability = self.rate > 1

    # def __repr__(self):
    #     return f'{self.profitability}'


@timeit
def bollinger(*args, **kwargs):
    stock = Stock(*args, **kwargs)
    for signal_begin in stock:
        if signal_begin.is_buy_start():
            for signal_end in stock.signal_iterator:
                if signal_end.is_buy_stop():
                    stock.make_transaction(signal_begin, signal_end)
                    break
    return stock


class Window:
    def __init__(self, transaction):
        self.data = transaction.history[[
            'open', 'avg', 'lower', 'upper']].reset_index(drop=True)
        self.data.rename(columns={'open': 'price'}, inplace=True)

    def normalize(self):
        self.data /= self.data.iloc[-1].price

    def draw(self):
        fig = go.Figure()
        for row in self.data:
            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=self.data[row],
                mode='lines',
                name=row
            ))
        fig.show()
