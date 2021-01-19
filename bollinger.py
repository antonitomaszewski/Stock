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
        self.money = transaction.money_output

    def get_history(self, signal_begin):
        return self.stock_df.loc[signal_begin.index - self.n + 1:
                                 signal_begin.index]

    def get_data_for_ML(self):
        return pd.DataFrame(columns=['Result'] + list(range(self.n)),
                            data=([transaction.profitability]
                                  + list(transaction.history.open /
                                         transaction.history.open.iloc[-1])
                                  for transaction in self.transactions)
                            )


class Signal:
    """
    Represents single row of Stock data
    """
    index = property()

    def __init__(self, row):
        self.row = row

    @index.getter
    def index(self):
        return self.row.index[0]

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
                 money_input, commission):
        self.history = history
        self.signal_begin = signal_begin
        self.signal_end = signal_end
        self.price_begin = signal_begin.get_price()
        self.price_end = signal_end.get_price()
        self.money_input = money_input
        self.money_output = commission * money_input * \
            self.price_end / self.price_begin
        self.profitability = self.money_input < self.money_output

    # def __repr__(self):
    #     return f'{self.profitability}'


@timeit
def bollinger(*args, **kwargs):
    stock = Stock(*args, **kwargs)
    stock_iterator = iter(stock)
    while True:
        try:
            signal_begin = next(stock_iterator)
        except StopIteration:
            return stock
        if signal_begin.is_buy_start():
            while True:
                try:
                    signal_end = next(stock_iterator)
                except StopIteration:
                    return stock
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
